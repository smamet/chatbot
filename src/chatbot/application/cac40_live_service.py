from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.application.integration_service import IntegrationService
from chatbot.application.tenant_service import TenantService
from chatbot.cac40.config import Cac40Config, public_config_snapshot
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.ig_connector import IgConnector
from chatbot.cac40.ig_ohlc import ig_config_from_connector
from chatbot.cac40.scheduler import LiveScheduler
from chatbot.config.settings import Settings
from chatbot.domain.models.integration import IntegrationType

logger = logging.getLogger(__name__)

LIVE_MODES = frozenset({"off", "paper", "live"})
LIVE_CYCLE_SECONDS = 900
# Run shortly after each 15m candle close so OHLC is available (Paris market clock).
LIVE_CYCLE_OFFSET_SECONDS = 15
LIVE_CYCLE_TZ = "Europe/Paris"

_LOCK = threading.Lock()
_SCHEDULERS: dict[str, LiveScheduler] = {}
_SCHEDULER_HASH: dict[str, str] = {}
# Last completed candle-close slot key per slug (e.g. "2026-07-21T12:00:00+02:00").
_LAST_CYCLE_SLOT: dict[str, str] = {}


def live_cycle_slot_key(now: datetime | None = None) -> str:
    """
    Wall-clock 15m slot aligned to candle closes + offset.

    Slots fire at :00:15, :15:15, :30:15, :45:15 in ``LIVE_CYCLE_TZ``.
    The key is the candle-close timestamp (e.g. 12:00 for the 12:00:15 run).
    """
    tz = ZoneInfo(LIVE_CYCLE_TZ)
    current = (now or datetime.now(tz)).astimezone(tz)
    adjusted = current - timedelta(seconds=LIVE_CYCLE_OFFSET_SECONDS)
    minute = (adjusted.minute // 15) * 15
    slot = adjusted.replace(minute=minute, second=0, microsecond=0)
    return slot.isoformat()


def _remember_cycle_slot(slug: str, slot: str | None = None) -> str:
    key = slot or live_cycle_slot_key()
    _LAST_CYCLE_SLOT[slug] = key
    return key


def _is_cycle_due_for_slot(settings: Settings, slug: str, slot: str) -> bool:
    """True once per candle slot. Always re-reads status for cross-process safety."""
    status = read_live_status(settings, slug)
    disk = str(status.get("last_cycle_slot") or "").strip() or None
    if disk:
        _LAST_CYCLE_SLOT[slug] = disk
    last = disk or _LAST_CYCLE_SLOT.get(slug)
    if last == slot:
        return False
    if last is None:
        # First arm / empty status: wait for the *next* slot (avoid mid-candle fire).
        _LAST_CYCLE_SLOT[slug] = slot
        write_live_status(
            settings,
            slug,
            {
                **status,
                "last_cycle_slot": slot,
                "awaiting_next_slot": True,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return False
    return True


@contextmanager
def live_cycle_lock(settings: Settings, slug: str, *, blocking: bool = False):
    """
    Cross-process lock so worker + manual Run cycle now cannot overlap.

    Yields True when the lock was acquired, False when non-blocking and busy.
    """
    import fcntl

    path = live_dir(settings, slug) / ".cycle.lock"
    fh = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(fh.fileno(), flags)
            acquired = True
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        if acquired:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        fh.close()


def _resolve_selected_ig_connectors(
    conn_svc: ConnectorService,
    tenant_id: int,
    live_cfg: dict[str, Any],
) -> tuple[list[tuple[int, dict[str, Any]]], list[str], str | None]:
    """
    Resolve configured IG connectors. Never silently picks an unselected account.

    Returns (connectors, warnings, error_code).
    """
    selected_ids = [int(i) for i in (live_cfg.get("ig_connector_ids") or [])]
    connectors: list[tuple[int, dict[str, Any]]] = []
    warnings: list[str] = []
    if not selected_ids:
        return [], warnings, "no_ig_connector"
    for cid in selected_ids:
        connector = conn_svc.get_ig_by_id(tenant_id, cid, active_only=True)
        if connector is None:
            warnings.append(f"dropped inactive/missing connector {cid}")
            continue
        connectors.append((cid, dict(connector.config)))
    if not connectors:
        return [], warnings, "no_ig_connector"
    return connectors, warnings, None


def live_dir(settings: Settings, slug: str) -> Path:
    path = Path(settings.data_root) / "cac40" / slug / "live"
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_config_path(settings: Settings, slug: str) -> Path:
    return live_dir(settings, slug) / "live_config.json"


def live_status_path(settings: Settings, slug: str) -> Path:
    return live_dir(settings, slug) / "status.json"


def live_state_path(settings: Settings, slug: str) -> Path:
    return live_dir(settings, slug) / "state.json"


def live_decisions_path(settings: Settings, slug: str) -> Path:
    return live_dir(settings, slug) / "decisions_log.json"


def live_journal_dir(settings: Settings, slug: str) -> Path:
    path = live_dir(settings, slug) / "journal"
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_worker_status_path(settings: Settings) -> Path:
    return Path(settings.data_root) / "cac40" / "live_worker_status.json"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def default_live_config() -> dict[str, Any]:
    cfg = Cac40Config().public_snapshot()
    return {
        "mode": "off",
        "ig_connector_ids": [],
        "strategy": {
            k: cfg[k]
            for k in (
                "max_open_positions",
                "order_size",
                "spread_points",
                "prevent_loss_exits",
                "llm_trigger_mode",
                "llm_level_band_points",
                "llm_every_n",
                "llm_every_unit",
                "llm_temperature",
                "lookback_15m",
                "lookback_1h",
                "lookback_1d",
                "warmup_bars",
                "chart_show_rsi",
                "chart_show_pivots",
                "chart_pivot_period",
            )
            if k in cfg
        },
    }


def load_live_config(settings: Settings, slug: str) -> dict[str, Any]:
    raw = _read_json(live_config_path(settings, slug), default=None)
    base = default_live_config()
    if not isinstance(raw, dict):
        return base
    mode = str(raw.get("mode") or "off").strip().lower()
    if mode not in LIVE_MODES:
        mode = "off"
    ids: list[int] = []
    for item in raw.get("ig_connector_ids") or []:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    strategy = dict(base["strategy"])
    incoming = raw.get("strategy") if isinstance(raw.get("strategy"), dict) else raw
    for key in list(strategy.keys()):
        if key in incoming and incoming[key] is not None:
            strategy[key] = incoming[key]
    return {"mode": mode, "ig_connector_ids": ids, "strategy": strategy}


def resolve_cac40_trading_banner(
    session: Session,
    settings: Settings,
    *,
    tenant_id: int,
    slug: str,
    allowed_integrations: list[str] | tuple[str, ...] | None,
) -> dict[str, Any] | None:
    """
    Topbar indicator when CAC40 Backtest is active for this bot.

    Returns None when the integration is inactive/disallowed; otherwise
    ``{"active": True, "mode": "off"|"paper"|"live", "slug": ...}``.
    """
    from chatbot.domain.models.integration_schema import is_integration_allowed

    if not is_integration_allowed(
        allowed_integrations, IntegrationType.CAC40_BACKTEST.value
    ):
        return None
    active = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
        tenant_id, type=IntegrationType.CAC40_BACKTEST
    )
    if active is None:
        return None
    mode = load_live_config(settings, slug)["mode"]
    if mode not in LIVE_MODES:
        mode = "off"
    return {"active": True, "mode": mode, "slug": slug}


def save_live_config(settings: Settings, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = load_live_config(settings, slug)
    mode = str(payload.get("mode") or current["mode"]).strip().lower()
    if mode not in LIVE_MODES:
        mode = current["mode"]
    ids: list[int] = []
    for item in payload.get("ig_connector_ids", current["ig_connector_ids"]) or []:
        try:
            cid = int(item)
        except (TypeError, ValueError):
            continue
        if cid not in ids:
            ids.append(cid)
    strategy = dict(current["strategy"])
    incoming = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else payload
    for key in list(strategy.keys()):
        if key in incoming and incoming[key] is not None:
            strategy[key] = incoming[key]
    # Normalize bools / numbers via Cac40Config
    cfg = Cac40Config.from_dict(strategy)
    every_n, every_unit, every_bars = Cac40Config.llm_rate_from_form(
        every_n=int(cfg.llm_every_n or 6), unit=str(cfg.llm_every_unit or "1h")
    )
    strategy = {
        "max_open_positions": int(cfg.max_open_positions),
        "order_size": float(cfg.order_size),
        "spread_points": float(cfg.spread_points),
        "prevent_loss_exits": bool(cfg.prevent_loss_exits),
        "llm_trigger_mode": str(cfg.llm_trigger_mode or "levels"),
        "llm_level_band_points": float(cfg.llm_level_band_points or 15.0),
        "llm_every_n": every_n,
        "llm_every_unit": every_unit,
        "llm_every_bars": every_bars,
        "llm_temperature": max(0.0, min(1.0, float(cfg.llm_temperature or 0.0))),
        "lookback_15m": max(1, int(cfg.lookback_15m)),
        "lookback_1h": max(1, int(cfg.lookback_1h)),
        "lookback_1d": max(1, int(cfg.lookback_1d)),
        "warmup_bars": max(2, int(cfg.warmup_bars)),
        "chart_show_rsi": bool(cfg.chart_show_rsi),
        "chart_show_pivots": bool(cfg.chart_show_pivots),
        "chart_pivot_period": str(cfg.chart_pivot_period or "D"),
    }
    saved = {"mode": mode, "ig_connector_ids": ids, "strategy": strategy}
    _write_json(live_config_path(settings, slug), saved)
    return saved


def set_live_mode(
    settings: Settings,
    slug: str,
    mode: str,
    *,
    session: Session | None = None,
    tenant_id: int | None = None,
    ig_connector_ids: list[int] | None = None,
) -> dict[str, Any]:
    mode = str(mode or "").strip().lower()
    if mode not in LIVE_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected off|paper|live")
    cfg = load_live_config(settings, slug)
    if ig_connector_ids is not None:
        ids: list[int] = []
        for item in ig_connector_ids:
            try:
                cid = int(item)
            except (TypeError, ValueError):
                continue
            if cid not in ids:
                ids.append(cid)
        cfg["ig_connector_ids"] = ids
    if mode == "live":
        if not cfg["ig_connector_ids"]:
            raise ValueError("Select at least one IG connector before arming Live.")
        if session is not None and tenant_id is not None:
            svc = ConnectorService(SqlAlchemyConnectorRepository(session))
            active_ids = {
                c.id for c in svc.list_ig(tenant_id, active_only=True)
            }
            usable = [i for i in cfg["ig_connector_ids"] if i in active_ids]
            if not usable:
                raise ValueError(
                    "No selected IG connector is active. Activate an IG connector first."
                )
            if usable != cfg["ig_connector_ids"]:
                cfg["ig_connector_ids"] = usable
    cfg["mode"] = mode
    return save_live_config(settings, slug, cfg)


def read_live_status(settings: Settings, slug: str) -> dict[str, Any]:
    raw = _read_json(live_status_path(settings, slug), default={}) or {}
    return raw if isinstance(raw, dict) else {}


def write_live_status(settings: Settings, slug: str, payload: dict[str, Any]) -> None:
    _write_json(live_status_path(settings, slug), payload)


def read_live_worker_status(settings: Settings) -> dict[str, Any]:
    raw = _read_json(live_worker_status_path(settings), default={}) or {}
    return raw if isinstance(raw, dict) else {}


def resolve_primary_ig_config(
    settings: Settings,
    slug: str,
    *,
    session: Session,
    tenant_id: int,
) -> dict[str, Any] | None:
    """Prefer live-config primary connector; fall back to first active IG."""
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    live_cfg = load_live_config(settings, slug)
    for cid in live_cfg.get("ig_connector_ids") or []:
        connector = svc.get_ig_by_id(tenant_id, int(cid), active_only=True)
        if connector is not None:
            return dict(connector.config)
    return svc.get_ig_config(tenant_id)


def _build_cac40_config(
    *,
    live_cfg: dict[str, Any],
    integ_cfg: dict[str, Any],
    primary_ig: dict[str, Any],
    tenant_slug: str,
    gemini_model: str,
) -> Cac40Config:
    strategy = dict(live_cfg.get("strategy") or {})
    ig_base = ig_config_from_connector(primary_ig)
    merged = {
        **strategy,
        "symbol": str(integ_cfg.get("symbol") or "CAC40"),
        "epic": str(primary_ig.get("epic") or integ_cfg.get("epic") or ig_base.epic),
        "ig_api_key": ig_base.ig_api_key,
        "ig_username": ig_base.ig_username,
        "ig_password": ig_base.ig_password,
        "ig_account_id": ig_base.ig_account_id,
        "ig_acc_type": ig_base.ig_acc_type,
        "fundmanager_url": str(integ_cfg.get("fundmanager_url") or ""),
        "fundmanager_token": str(integ_cfg.get("fundmanager_token") or ""),
        "bot_id": tenant_slug,
        "llm_mode": "live",
        "gemini_model": gemini_model,
    }
    if integ_cfg.get("max_open_positions") not in (None, "") and "max_open_positions" not in strategy:
        try:
            merged["max_open_positions"] = int(integ_cfg["max_open_positions"])
        except (TypeError, ValueError):
            pass
    cfg = Cac40Config.from_dict(merged)
    cfg.llm_mode = "live"
    cfg.llm_every_bars = cfg.resolve_llm_every_bars()
    return cfg


def _config_hash(
    *,
    mode: str,
    connector_ids: list[int],
    cfg: Cac40Config,
) -> str:
    payload = {
        "mode": mode,
        "ids": connector_ids,
        "strategy": public_config_snapshot(cfg.to_dict()),
        "epic": cfg.epic,
        "ig_account_id": cfg.ig_account_id,
        "ig_acc_type": cfg.ig_acc_type,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _append_decision(settings: Settings, slug: str, payload: dict[str, Any]) -> None:
    path = live_decisions_path(settings, slug)
    entries = _read_json(path, default=[]) or []
    if not isinstance(entries, list):
        entries = []
    if payload.get("decision") or payload.get("charts_rel"):
        entries.append(
            {
                "ts": payload.get("ts"),
                "decision": payload.get("decision"),
                "executed": payload.get("executed") or [],
                "rejected": payload.get("rejected") or [],
                "charts_rel": payload.get("charts_rel") or "",
                "chart_files": payload.get("chart_files") or [],
                "pnl": payload.get("pnl"),
                "skipped": payload.get("skipped"),
                "llm_trigger": payload.get("llm_trigger") or [],
                "mirror": payload.get("mirror") or [],
            }
        )
        _write_json(path, entries[-500:])


def clear_live_history(settings: Settings, slug: str, *, mode: str | None = None) -> None:
    """Clear paper journal/state. Blocked when mode is live."""
    cfg = load_live_config(settings, slug)
    current = mode or cfg["mode"]
    if current == "live":
        raise ValueError("Cannot clear history while Live mode is armed.")
    state = live_state_path(settings, slug)
    decisions = live_decisions_path(settings, slug)
    if state.exists():
        state.unlink()
    if decisions.exists():
        decisions.unlink()
    # Drop mirrored dealId books so a later Live arm does not cancel stale IG orders.
    books = live_dir(settings, slug) / "order_books"
    if books.is_dir():
        for child in books.iterdir():
            if child.is_file():
                child.unlink()
    journal = live_journal_dir(settings, slug)
    for child in journal.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_dir():
                    nested.rmdir()
            child.rmdir()
    with _LOCK:
        sched = _SCHEDULERS.pop(slug, None)
        _SCHEDULER_HASH.pop(slug, None)
        _LAST_CYCLE_SLOT.pop(slug, None)
    if sched is not None:
        try:
            sched.close()
        except Exception:
            pass


def get_live_report(settings: Settings, slug: str) -> dict[str, Any]:
    """Build a run-like report payload for the live results page."""
    live_cfg = load_live_config(settings, slug)
    status = read_live_status(settings, slug)
    state_raw = _read_json(live_state_path(settings, slug), default={}) or {}
    root = live_dir(settings, slug)
    decisions_raw = _read_json(live_decisions_path(settings, slug), default=[]) or []
    if not isinstance(decisions_raw, list):
        decisions_raw = []

    decisions: list[dict[str, Any]] = []
    for entry in decisions_raw:
        charts_rel = str(entry.get("charts_rel") or "").strip().replace("\\", "/")
        chart_files = list(entry.get("chart_files") or [])
        charts = []
        if charts_rel.startswith("journal/") and chart_files:
            parts = charts_rel.split("/")
            if len(parts) >= 3:
                cycle = parts[1]
                for name in chart_files:
                    tf = name.removeprefix("chart_").removesuffix(".png")
                    charts.append(
                        {
                            "tf": tf,
                            "file": name,
                            "url": (
                                f"/dashboard/bots/{slug}/cac40/live/charts/"
                                f"{cycle}/{name}"
                            ),
                        }
                    )
        dec = entry.get("decision") or {}
        analysis = dec.get("analysis") or {}
        decisions.append(
            {
                **entry,
                "chart_files": chart_files,
                "charts": charts,
                "bias": analysis.get("bias"),
                "support": analysis.get("support"),
                "resistance": analysis.get("resistance"),
                "actions": dec.get("actions") or [],
            }
        )
    decisions.reverse()

    positions = state_raw.get("positions") or []
    working = state_raw.get("working_orders") or []
    closed = state_raw.get("closed_trades") or []
    net_upl = sum(float(p.get("upl") or 0) for p in positions if isinstance(p, dict))
    realized = float(state_raw.get("realized_session") or 0)
    mode = live_cfg["mode"]
    return {
        "run_id": "live",
        "live": True,
        "path": str(root),
        "state": {
            "status": status.get("last_status") or mode,
            "progress": None,
            "current_bar": status.get("cycles") or 0,
            "total_bars": None,
            "error": status.get("error") or state_raw.get("error"),
        },
        "report": {
            "config": {
                **(live_cfg.get("strategy") or {}),
                "mode": mode,
                "ig_connector_ids": live_cfg.get("ig_connector_ids") or [],
            },
            "final_equity": realized + net_upl + float(state_raw.get("cash") or 0),
            "realized_pnl": realized,
            "max_drawdown": None,
            "trades": len(closed),
            "winrate": None,
            "decisions_count": len(decisions_raw),
            "closed_trades": [
                {
                    "id": t.get("id"),
                    "side": t.get("side"),
                    "role": t.get("role"),
                    "entry": t.get("entry"),
                    "exit": t.get("exit"),
                    "pnl": t.get("realized_pnl"),
                    "bars_held": t.get("bars_held"),
                }
                for t in closed
                if isinstance(t, dict)
            ],
            "summary": {
                "cycles": status.get("cycles") or len(decisions_raw),
                "open_legs": len(positions),
                "realized": realized,
                "net_upl": net_upl,
            },
            "book": {
                "positions": positions,
                "working_orders": working,
                "phase": state_raw.get("phase") or "Flat",
                "last_price": state_raw.get("last_price"),
            },
        },
        "decisions": decisions,
        "mode": mode,
        "live_status": status,
        "pnl": {
            "net_upl": net_upl,
            "realized_session": realized,
            "legs_count": len(positions),
            "working_orders_count": len(working),
        },
    }


def resolve_live_chart_file(
    settings: Settings, slug: str, cycle: str, filename: str
) -> Path | None:
    if ".." in cycle or "/" in cycle or "\\" in cycle:
        return None
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    if not filename.startswith("chart_") or not filename.endswith(".png"):
        return None
    path = live_journal_dir(settings, slug) / cycle / "charts" / filename
    return path if path.is_file() else None


def _get_or_build_scheduler(
    *,
    settings: Settings,
    slug: str,
    live_cfg: dict[str, Any],
    cfg: Cac40Config,
    api_key: str,
    connectors: list[tuple[int, dict[str, Any]]],
    tenant_id: int,
    session_factory,
) -> LiveScheduler:
    mode = live_cfg["mode"]
    dry_run = mode != "live"
    ids = [cid for cid, _ in connectors]
    digest = _config_hash(mode=mode, connector_ids=ids, cfg=cfg)
    with _LOCK:
        existing = _SCHEDULERS.get(slug)
        if existing is not None and _SCHEDULER_HASH.get(slug) == digest:
            return existing
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
            _SCHEDULERS.pop(slug, None)

    primary_cfg = connectors[0][1]
    order_connectors: list[tuple[int, IgConnector]] = []
    sched = LiveScheduler(
        cfg,
        api_key=api_key,
        journal_dir=live_journal_dir(settings, slug),
        dry_run=dry_run,
        sleep_seconds=LIVE_CYCLE_SECONDS,
        tenant_id=tenant_id,
        session_factory=session_factory,
        orders_dir=live_dir(settings, slug) / "order_books",
    )
    # Restore ledger
    state = _read_json(live_state_path(settings, slug), default=None)
    if isinstance(state, dict) and state:
        sched.ig.ledger = HedgeLedger.from_state_dict(cfg, state)

    order_connectors.append((connectors[0][0], sched.ig))
    for cid, conn_cfg in connectors[1:]:
        secondary_cfg = ig_config_from_connector(conn_cfg)
        secondary = IgConnector(secondary_cfg, dry_run=dry_run)
        order_connectors.append((cid, secondary))
    sched.order_connectors = order_connectors

    with _LOCK:
        _SCHEDULERS[slug] = sched
        _SCHEDULER_HASH[slug] = digest
    return sched


def run_live_cycle_now(
    session: Session,
    settings: Settings,
    slug: str,
    *,
    session_factory=None,
) -> dict[str, Any]:
    """
    Force one live/paper cycle for a bot (ignores the candle clock).

    Returns {ok, message, error?, payload?}.
    """
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.interfaces.api.deps import _gemini_api_key

    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    tenant = tenant_svc.get_by_slug(slug)
    if tenant is None:
        return {"ok": False, "message": f"Bot {slug!r} not found.", "error": "not_found"}

    integ = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
        tenant.id, type=IntegrationType.CAC40_BACKTEST
    )
    if integ is None:
        return {
            "ok": False,
            "message": "CAC40 integration is not active.",
            "error": "no_integration",
        }

    live_cfg = load_live_config(settings, slug)
    mode = live_cfg["mode"]
    if mode == "off":
        return {
            "ok": False,
            "message": "Bot is Off — switch to Paper or Live first, then Run cycle now.",
            "error": "mode_off",
        }

    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    connectors, warnings, conn_err = _resolve_selected_ig_connectors(
        conn_svc, tenant.id, live_cfg
    )
    if conn_err:
        return {
            "ok": False,
            "message": "No selected active IG connector — check accounts, Save live config, retry.",
            "error": conn_err,
        }

    if session_factory is None:
        try:
            from chatbot.adapters.persistence.engine import session_factory as make_factory

            session_factory = make_factory(session.get_bind())
        except Exception:
            session_factory = None

    integ_cfg = dict(integ.config or {})
    gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
    cfg = _build_cac40_config(
        live_cfg=live_cfg,
        integ_cfg=integ_cfg,
        primary_ig=connectors[0][1],
        tenant_slug=slug,
        gemini_model=gemini_model,
    )
    api_key = _gemini_api_key(tenant, settings) or ""
    if not api_key:
        return {
            "ok": False,
            "message": "No Gemini API key (tenant or GEMINI_API_KEY) — cannot call the LLM.",
            "error": "no_gemini_key",
        }

    prev_status = read_live_status(settings, slug)
    prev_cycles = int(prev_status.get("cycles") or 0)
    with live_cycle_lock(settings, slug, blocking=False) as acquired:
        if not acquired:
            return {
                "ok": False,
                "message": "A cycle is already running for this bot — try again in a moment.",
                "error": "cycle_busy",
            }
        try:
            sched = _get_or_build_scheduler(
                settings=settings,
                slug=slug,
                live_cfg=live_cfg,
                cfg=cfg,
                api_key=api_key,
                connectors=connectors,
                tenant_id=tenant.id,
                session_factory=session_factory,
            )
            payload = sched.run_once()
            _write_json(live_state_path(settings, slug), sched.ig.ledger.to_state_dict())
            _append_decision(settings, slug, payload)
            slot = _remember_cycle_slot(slug)
            pnl = sched.ig.ledger.pnl_payload()
            write_live_status(
                settings,
                slug,
                {
                    "mode": mode,
                    "dry_run": mode != "live",
                    "ig_connector_ids": [c[0] for c in connectors],
                    "last_cycle_at": payload.get("ts"),
                    "last_cycle_slot": slot,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "last_status": "ok",
                    "error": None,
                    "warnings": warnings,
                    "mirror": payload.get("mirror") or [],
                    "open_legs": sched.ig.ledger.legs_count(),
                    "working_orders": len(sched.ig.ledger.working_orders),
                    "pnl": pnl,
                    "last_decision": sched._last_decision_summary,
                    "skipped_llm": bool(payload.get("skipped")),
                    "cycles": prev_cycles + 1,
                    "trigger": "manual",
                },
            )
            # Touch global worker status so the UI heartbeat stays green.
            _write_json(
                live_worker_status_path(settings),
                {
                    "ok": True,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "poll_seconds": settings.cac40_live_poll_seconds,
                    "tenants_ok": 1,
                    "tenants_failed": 0,
                    "tenants_skipped": 0,
                    "logs": [f"{slug}: manual cycle ok mode={mode}"],
                    "trigger": "manual",
                },
            )
            llm_bit = (
                "LLM skipped (trigger quiet)"
                if payload.get("skipped")
                else (
                    f"LLM ran · executed={len(payload.get('executed') or [])} "
                    f"rejected={len(payload.get('rejected') or [])}"
                )
            )
            msg = (
                f"Cycle OK ({mode}) · {llm_bit} · "
                f"legs={pnl.get('legs_count', 0)} · "
                f"orders={pnl.get('working_orders_count', 0)} · "
                f"realized={float(pnl.get('realized_session') or 0):.2f}"
            )
            if warnings:
                msg += f" · note: {'; '.join(warnings)}"
            return {"ok": True, "message": msg, "error": None, "payload": payload}
        except Exception as exc:
            logger.exception("Manual live cycle failed for %s", slug)
            write_live_status(
                settings,
                slug,
                {
                    "mode": mode,
                    "error": str(exc),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "last_status": "error",
                    "warnings": warnings,
                    "cycles": prev_cycles,
                    "trigger": "manual",
                },
            )
            return {
                "ok": False,
                "message": f"Cycle failed: {exc}",
                "error": str(exc),
            }


def run_due_live_cycles(session: Session, settings: Settings) -> list[str]:
    """One worker poll: run a live cycle for each armed bot due on the candle clock."""
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.interfaces.api.deps import _gemini_api_key

    started = datetime.now(timezone.utc).isoformat()
    logs: list[str] = []
    ok = 0
    failed = 0
    skipped = 0
    slot = live_cycle_slot_key()

    integ_repo = SqlAlchemyIntegrationRepository(session)
    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))

    try:
        from chatbot.adapters.persistence.engine import session_factory as make_factory

        factory = make_factory(session.get_bind())
    except Exception:
        factory = None

    for integration in integ_repo.list_active_by_type(IntegrationType.CAC40_BACKTEST):
        tenant = tenant_svc.get_by_id(integration.tenant_id)
        if tenant is None:
            skipped += 1
            continue
        slug = tenant.slug
        live_cfg = load_live_config(settings, slug)
        mode = live_cfg["mode"]
        if mode == "off":
            skipped += 1
            continue

        selected_ids = [int(i) for i in (live_cfg.get("ig_connector_ids") or [])]
        connectors, warnings, conn_err = _resolve_selected_ig_connectors(
            conn_svc, tenant.id, live_cfg
        )
        if selected_ids and connectors and len(connectors) != len(selected_ids):
            live_cfg["ig_connector_ids"] = [c[0] for c in connectors]
            save_live_config(settings, slug, live_cfg)
        if conn_err:
            skipped += 1
            write_live_status(
                settings,
                slug,
                {
                    "mode": mode,
                    "error": "no_ig_connector",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logs.append(f"{slug}: skip (no selected IG connector)")
            continue

        integ_cfg = dict(integration.config or {})
        gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
        cfg = _build_cac40_config(
            live_cfg=live_cfg,
            integ_cfg=integ_cfg,
            primary_ig=connectors[0][1],
            tenant_slug=slug,
            gemini_model=gemini_model,
        )
        api_key = _gemini_api_key(tenant, settings) or ""
        prev_status = read_live_status(settings, slug)
        prev_cycles = int(prev_status.get("cycles") or 0)
        with live_cycle_lock(settings, slug, blocking=False) as acquired:
            if not acquired:
                skipped += 1
                logs.append(f"{slug}: skip (cycle lock busy)")
                continue
            if not _is_cycle_due_for_slot(settings, slug, slot):
                skipped += 1
                continue
            try:
                sched = _get_or_build_scheduler(
                    settings=settings,
                    slug=slug,
                    live_cfg=live_cfg,
                    cfg=cfg,
                    api_key=api_key,
                    connectors=connectors,
                    tenant_id=tenant.id,
                    session_factory=factory,
                )
                payload = sched.run_once()
                _write_json(live_state_path(settings, slug), sched.ig.ledger.to_state_dict())
                _append_decision(settings, slug, payload)
                _remember_cycle_slot(slug, slot)
                write_live_status(
                    settings,
                    slug,
                    {
                        "mode": mode,
                        "dry_run": mode != "live",
                        "ig_connector_ids": [c[0] for c in connectors],
                        "last_cycle_at": payload.get("ts"),
                        "last_cycle_slot": slot,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "last_status": "ok",
                        "error": None,
                        "warnings": warnings,
                        "mirror": payload.get("mirror") or [],
                        "open_legs": sched.ig.ledger.legs_count(),
                        "working_orders": len(sched.ig.ledger.working_orders),
                        "pnl": sched.ig.ledger.pnl_payload(),
                        "last_decision": sched._last_decision_summary,
                        "skipped_llm": bool(payload.get("skipped")),
                        "cycles": prev_cycles + 1,
                    },
                )
                ok += 1
                logs.append(f"{slug}: cycle ok mode={mode}")
            except Exception as exc:
                failed += 1
                logger.exception("Live cycle failed for %s", slug)
                write_live_status(
                    settings,
                    slug,
                    {
                        "mode": mode,
                        "error": str(exc),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "last_status": "error",
                        "warnings": warnings,
                        "cycles": prev_cycles,
                    },
                )
                logs.append(f"{slug}: error {exc}")

    finished = datetime.now(timezone.utc).isoformat()
    _write_json(
        live_worker_status_path(settings),
        {
            "ok": failed == 0,
            "started_at": started,
            "finished_at": finished,
            "poll_seconds": settings.cac40_live_poll_seconds,
            "tenants_ok": ok,
            "tenants_failed": failed,
            "tenants_skipped": skipped,
            "logs": logs[-50:],
        },
    )
    return logs
