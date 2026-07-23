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
from chatbot.application.cac40_backtest_service import default_ohlc_path
from chatbot.cac40.config import Cac40Config, public_config_snapshot
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.ig_connector import IgConnector
from chatbot.cac40.ig_ohlc import ig_config_from_connector
from chatbot.cac40.live_ohlc_feed import prepare_live_ohlc_feed
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


def llm_schedule_path(settings: Settings, slug: str) -> Path:
    return live_dir(settings, slug) / "llm_schedule.json"


def load_llm_schedule(settings: Settings, slug: str) -> dict[str, Any]:
    """
    Persisted Fixed-rate gate. Seeds last_llm_at from the newest LLM cycle if missing
    so a deploy/restart does not immediately re-fire Gemini.
    """
    raw = _read_json(llm_schedule_path(settings, slug), default={}) or {}
    if not isinstance(raw, dict):
        raw = {}
    last = str(raw.get("last_llm_at") or "").strip()
    if not last:
        for cycle in list_live_cycles(settings, slug, limit=1):
            ts = str(cycle.get("ts") or "").strip()
            if ts:
                last = ts
                break
    return {
        "last_llm_at": last or None,
        "every_bars": raw.get("every_bars"),
        "mode": raw.get("mode"),
    }


def save_llm_schedule(
    settings: Settings,
    slug: str,
    *,
    last_llm_at: datetime | str | None,
    every_bars: int | None = None,
    mode: str | None = None,
) -> None:
    if isinstance(last_llm_at, datetime):
        ts = last_llm_at.astimezone(timezone.utc).isoformat()
    else:
        ts = str(last_llm_at).strip() if last_llm_at else None
    payload = {
        "last_llm_at": ts,
        "every_bars": every_bars,
        "mode": mode,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(llm_schedule_path(settings, slug), payload)


def _apply_llm_schedule(sched: LiveScheduler, settings: Settings, slug: str) -> None:
    """Configure live interval spacing from disk (wall clock)."""
    sched.trigger.interval_clock = "wall"
    sched.trigger.every_bars = int(sched.config.resolve_llm_every_bars())
    sched.trigger.mode = str(sched.config.llm_trigger_mode or "levels")
    schedule = load_llm_schedule(settings, slug)
    raw_ts = schedule.get("last_llm_at")
    if raw_ts:
        try:
            parsed = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            sched.trigger.last_llm_at = parsed.astimezone(timezone.utc)
        except ValueError:
            sched.trigger.last_llm_at = None


def _persist_llm_schedule(sched: LiveScheduler, settings: Settings, slug: str) -> None:
    if sched.trigger.last_llm_at is None:
        return
    save_llm_schedule(
        settings,
        slug,
        last_llm_at=sched.trigger.last_llm_at,
        every_bars=int(sched.trigger.every_bars or 0) or None,
        mode=str(sched.trigger.mode or ""),
    )


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
                "min_exit_profit_points",
                "flatten_before_close",
                "flatten_lead_minutes",
                "market_close_paris",
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
        "min_exit_profit_points": max(0.0, float(cfg.min_exit_profit_points or 0)),
        "flatten_before_close": bool(cfg.flatten_before_close),
        "flatten_lead_minutes": max(1, int(cfg.flatten_lead_minutes or 30)),
        "market_close_paris": str(cfg.market_close_paris or "22:00"),
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
    saved = save_live_config(settings, slug, cfg)
    # Force GET /positions on the next cycle after arming paper/live.
    if mode in ("paper", "live"):
        with _LOCK:
            sched = _SCHEDULERS.get(slug)
            if sched is not None:
                sched.request_position_reconcile()
    return saved


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


def _normalize_decision_pnl(pnl: Any) -> dict[str, Any]:
    """Align live pnl_payload (realized_session) with backtest (realized) for run.html."""
    if not isinstance(pnl, dict):
        return {}
    out = dict(pnl)
    if "realized" not in out or out.get("realized") is None:
        if out.get("realized_session") is not None:
            out["realized"] = out["realized_session"]
    return out


def _book_from_snapshot(snapshot: Any) -> dict[str, Any]:
    """Compact book counts for the decision summary line."""
    if not isinstance(snapshot, dict):
        return {"positions": 0, "working_orders": 0}
    positions = snapshot.get("positions")
    working = snapshot.get("working_orders")
    if isinstance(positions, list):
        pos_n = len(positions)
    elif isinstance(positions, dict):
        pos_n = len(positions)
    else:
        pos_n = int(snapshot.get("legs_count") or 0)
    if isinstance(working, list):
        wo_n = len(working)
    elif isinstance(working, dict):
        wo_n = len(working)
    else:
        wo_n = int(snapshot.get("working_orders_count") or 0)
    return {
        "positions": pos_n,
        "working_orders": wo_n,
        "phase": snapshot.get("phase") or snapshot.get("book_phase"),
        "last_price": snapshot.get("last_price") or snapshot.get("mid"),
    }


def _append_decision(settings: Settings, slug: str, payload: dict[str, Any]) -> None:
    path = live_decisions_path(settings, slug)
    entries = _read_json(path, default=[]) or []
    if not isinstance(entries, list):
        entries = []
    if payload.get("decision") or payload.get("charts_rel"):
        snap = payload.get("snapshot") or {}
        entries.append(
            {
                "ts": payload.get("ts"),
                "decision": payload.get("decision"),
                "executed": payload.get("executed") or [],
                "rejected": payload.get("rejected") or [],
                "charts_rel": payload.get("charts_rel") or "",
                "chart_files": payload.get("chart_files") or [],
                "pnl": _normalize_decision_pnl(payload.get("pnl")),
                "skipped": payload.get("skipped"),
                "llm_trigger": payload.get("llm_trigger") or [],
                "mirror": payload.get("mirror") or [],
                "cycle_dir": payload.get("cycle_dir") or "",
                "book": _book_from_snapshot(snap),
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


def _live_chart_urls(
    slug: str, cycle: str, chart_files: list[str]
) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    for name in chart_files:
        tf = name.removeprefix("chart_").removesuffix(".png")
        charts.append(
            {
                "tf": tf,
                "file": name,
                "url": f"/dashboard/bots/{slug}/cac40/live/charts/{cycle}/{name}",
            }
        )
    return charts


def _decision_row_from_entry(
    entry: dict[str, Any], *, slug: str, journal_root: Path
) -> dict[str, Any]:
    """Map a cycle.json / decisions_log entry into run.html decision shape."""
    charts_rel = str(entry.get("charts_rel") or "").strip().replace("\\", "/")
    chart_files = list(entry.get("chart_files") or [])
    cycle = str(entry.get("cycle_dir") or "").strip()
    if not cycle and charts_rel.startswith("journal/"):
        parts = charts_rel.split("/")
        if len(parts) >= 2:
            cycle = parts[1]
    if cycle and not chart_files:
        chart_dir = journal_root / cycle / "charts"
        if chart_dir.is_dir():
            chart_files = sorted(p.name for p in chart_dir.glob("chart_*.png"))
    charts = _live_chart_urls(slug, cycle, chart_files) if cycle and chart_files else []
    dec = entry.get("decision") or {}
    if not isinstance(dec, dict):
        dec = {}
    analysis = dec.get("analysis") or {}
    book = entry.get("book")
    if not isinstance(book, dict) or (
        "positions" in book and not isinstance(book.get("positions"), int)
    ):
        book = _book_from_snapshot(entry.get("snapshot") or {})
        # If book still has list-shaped leftovers from a bad merge, coerce counts.
        if isinstance(book.get("positions"), list):
            book["positions"] = len(book["positions"])
        if isinstance(book.get("working_orders"), list):
            book["working_orders"] = len(book["working_orders"])
    return {
        **entry,
        "cycle_dir": cycle,
        "chart_files": chart_files,
        "charts": charts,
        "bias": analysis.get("bias"),
        "support": analysis.get("support"),
        "resistance": analysis.get("resistance"),
        "actions": dec.get("actions") or [],
        "pnl": _normalize_decision_pnl(entry.get("pnl")),
        "book": {
            "positions": int(book.get("positions") or 0),
            "working_orders": int(book.get("working_orders") or 0),
            "phase": book.get("phase"),
            "last_price": book.get("last_price"),
        },
    }


def _cycle_worth_showing(raw: dict[str, Any]) -> bool:
    """True for cycles that belong in the live results browser / index list."""
    if raw.get("decision") or raw.get("charts_rel") or raw.get("chart_files"):
        return True
    if raw.get("rejected") or raw.get("executed"):
        return True
    return False


def _cycle_id_from_entry(entry: dict[str, Any]) -> str:
    cycle = str(entry.get("cycle_dir") or "").strip()
    if cycle:
        return cycle
    charts_rel = str(entry.get("charts_rel") or "").strip().replace("\\", "/")
    if charts_rel.startswith("journal/"):
        parts = charts_rel.split("/")
        if len(parts) >= 2:
            return parts[1]
    return ""


def list_live_cycles(
    settings: Settings, slug: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Recent live/paper cycles with LLM/charts (newest first). Matches report rows."""
    journal = live_journal_dir(settings, slug)
    rows: list[dict[str, Any]] = []
    for cycle_json in journal.glob("*/cycle.json"):
        cycle_id = cycle_json.parent.name
        if cycle_id.startswith("."):
            continue
        raw = _read_json(cycle_json, default=None)
        if not isinstance(raw, dict) or not _cycle_worth_showing(raw):
            continue
        dec = raw.get("decision") or {}
        analysis = (dec.get("analysis") or {}) if isinstance(dec, dict) else {}
        chart_files = list(raw.get("chart_files") or [])
        if not chart_files:
            chart_dir = cycle_json.parent / "charts"
            if chart_dir.is_dir():
                chart_files = sorted(p.name for p in chart_dir.glob("chart_*.png"))
        rows.append(
            {
                "cycle_id": cycle_id,
                "ts": raw.get("ts"),
                "skipped": bool(raw.get("skipped")),
                "has_charts": bool(chart_files),
                "bias": analysis.get("bias"),
                "executed_count": len(raw.get("executed") or []),
                "rejected_count": len(raw.get("rejected") or []),
                "dry_run": bool(raw.get("dry_run")),
            }
        )
    rows.sort(key=lambda r: str(r.get("ts") or r.get("cycle_id") or ""), reverse=True)
    return rows[: max(1, limit)]


def _load_live_decision_entries(settings: Settings, slug: str) -> list[dict[str, Any]]:
    """Load from journal cycle.json; merge any older decisions_log rows not already covered."""
    journal = live_journal_dir(settings, slug)
    entries: list[dict[str, Any]] = []
    seen_cycles: set[str] = set()
    for cycle_json in sorted(journal.glob("*/cycle.json")):
        raw = _read_json(cycle_json, default=None)
        if not isinstance(raw, dict) or not _cycle_worth_showing(raw):
            continue
        if not raw.get("cycle_dir"):
            raw = {**raw, "cycle_dir": cycle_json.parent.name}
        cid = _cycle_id_from_entry(raw) or cycle_json.parent.name
        seen_cycles.add(cid)
        entries.append(raw)

    log_raw = _read_json(live_decisions_path(settings, slug), default=[]) or []
    if isinstance(log_raw, list):
        for entry in log_raw:
            if not isinstance(entry, dict) or not _cycle_worth_showing(entry):
                continue
            cid = _cycle_id_from_entry(entry)
            if cid and cid in seen_cycles:
                continue
            if cid:
                seen_cycles.add(cid)
            entries.append(entry)

    entries.sort(key=lambda e: str(e.get("ts") or _cycle_id_from_entry(e) or ""))
    out = [
        _decision_row_from_entry(e, slug=slug, journal_root=journal) for e in entries
    ]
    out.reverse()
    return out


def get_live_report(settings: Settings, slug: str) -> dict[str, Any]:
    """Build a run-like report payload for the live results page."""
    live_cfg = load_live_config(settings, slug)
    status = read_live_status(settings, slug)
    state_raw = _read_json(live_state_path(settings, slug), default={}) or {}
    root = live_dir(settings, slug)
    decisions = _load_live_decision_entries(settings, slug)

    positions = state_raw.get("positions") or []
    working = state_raw.get("working_orders") or []
    closed = state_raw.get("closed_trades") or []
    if not isinstance(positions, list):
        positions = list(positions.values()) if isinstance(positions, dict) else []
    if not isinstance(working, list):
        working = list(working.values()) if isinstance(working, dict) else []
    if not isinstance(closed, list):
        closed = []
    net_upl = sum(float(p.get("upl") or 0) for p in positions if isinstance(p, dict))
    realized = float(state_raw.get("realized_session") or 0)
    mode = live_cfg["mode"]
    llm_calls = sum(
        1
        for d in decisions
        if not d.get("skipped")
        and (d.get("decision") or d.get("charts") or d.get("chart_files"))
    )
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
            "decisions_count": len(decisions),
            "llm_calls_total": llm_calls,
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
                "cycles": status.get("cycles") or len(decisions),
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
            "realized": realized,
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


def _attach_local_ohlc_provider(
    sched: LiveScheduler, *, settings: Settings, slug: str
) -> None:
    """Inject CSV top-up + resample feed (dashboard live/paper path)."""
    path = default_ohlc_path(settings, slug)

    def _provider():
        return prepare_live_ohlc_feed(
            path,
            config=sched.config,
            connector=sched.ig,
            top_up=True,
        )

    sched.ohlc_provider = _provider


def _status_from_cycle_payload(
    payload: dict[str, Any],
    *,
    base_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Merge OHLC feed meta into live status fields."""
    feed = payload.get("ohlc_feed") if isinstance(payload, dict) else None
    feed = feed if isinstance(feed, dict) else {}
    warnings = list(base_warnings or [])
    for w in feed.get("warnings") or []:
        if w and w not in warnings:
            warnings.append(str(w))
    auto = payload.get("auto_flatten") if isinstance(payload.get("auto_flatten"), dict) else {}
    for err in auto.get("errors") or []:
        if err and str(err) not in warnings:
            warnings.append(str(err))
    clock = payload.get("market_clock") if isinstance(payload.get("market_clock"), dict) else {}
    allowance = feed.get("allowance") if isinstance(feed.get("allowance"), dict) else None
    err = feed.get("error")
    out: dict[str, Any] = {
        "warnings": warnings,
        "ohlc_last_bar": feed.get("last_bar_ts"),
        "ohlc_top_up_added": feed.get("top_up_added"),
        "ohlc_stale": bool(feed.get("stale")),
        "ig_price_allowance": allowance,
        "flatten_now": bool(clock.get("flatten_now")),
        "flatten_reason": clock.get("reason") or "",
        "auto_flatten": auto or None,
    }
    if err:
        out["error"] = str(err)
        out["last_status"] = "error" if feed.get("skip_llm") else "ok"
    if auto.get("errors"):
        out["last_status"] = "error"
        out["error"] = out.get("error") or "auto_flatten_partial_failure"
    return out


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
            _attach_local_ohlc_provider(existing, settings=settings, slug=slug)
            return existing
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
            _SCHEDULERS.pop(slug, None)

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
    _attach_local_ohlc_provider(sched, settings=settings, slug=slug)
    _apply_llm_schedule(sched, settings, slug)

    with _LOCK:
        _SCHEDULERS[slug] = sched
        _SCHEDULER_HASH[slug] = digest
    return sched


def _parse_ig_working_order_row(
    row: dict[str, Any], *, epic: str
) -> dict[str, Any] | None:
    """Normalize an IG working-order payload into side/type/level/size/deal_id."""
    from chatbot.cac40.models import OrderType, Side

    data = row.get("workingOrderData") if isinstance(row.get("workingOrderData"), dict) else row
    if not isinstance(data, dict):
        return None
    row_epic = str(data.get("epic") or "").strip()
    if epic and row_epic and row_epic != epic:
        return None
    deal_id = str(data.get("dealId") or "").strip()
    if not deal_id:
        return None
    direction = str(data.get("direction") or "").upper()
    if direction not in ("BUY", "SELL"):
        return None
    raw_type = str(data.get("orderType") or data.get("type") or "LIMIT").upper()
    otype = OrderType.STOP if "STOP" in raw_type else OrderType.LIMIT
    try:
        level = float(
            data.get("orderLevel")
            if data.get("orderLevel") is not None
            else data.get("level")
            or 0
        )
    except (TypeError, ValueError):
        level = 0.0
    try:
        size = float(
            data.get("orderSize")
            if data.get("orderSize") is not None
            else data.get("size")
            or 0
        )
    except (TypeError, ValueError):
        size = 0.0
    if level <= 0 or size <= 0:
        return None
    return {
        "deal_id": deal_id,
        "side": Side.BUY if direction == "BUY" else Side.SELL,
        "type": otype,
        "level": level,
        "size": size,
        "epic": row_epic,
    }


def adopt_ig_snapshot_into_ledger(
    ledger: HedgeLedger,
    *,
    positions: list[dict[str, Any]],
    working_orders: list[dict[str, Any]],
    epic: str = "",
) -> dict[str, Any]:
    """
    Additive import of IG positions/orders into a local ledger (idempotent by deal_id).

    Returns ``{imported_positions, imported_orders, skipped, warnings, order_book}``
    where ``order_book`` maps new local order ids → IG dealId.
    """
    from chatbot.cac40.models import (
        LegRole,
        OrderPurpose,
        OrderType,
        Side,
        WorkingOrder,
    )

    want = (epic or "").strip()
    known_pos = {
        (p.deal_id or "").strip()
        for p in ledger.positions.values()
        if (p.deal_id or "").strip()
    }
    known_wo = {
        (o.deal_id or "").strip()
        for o in ledger.working_orders.values()
        if (o.deal_id or "").strip()
    }

    imported_positions: list[dict[str, Any]] = []
    imported_orders: list[dict[str, Any]] = []
    skipped: list[str] = []
    warnings: list[str] = []
    order_book: dict[str, str] = {}

    # --- positions ---
    for row in positions:
        if not isinstance(row, dict):
            continue
        deal_id = str(row.get("deal_id") or "").strip()
        if not deal_id:
            skipped.append("position:missing_deal_id")
            continue
        if deal_id in known_pos:
            skipped.append(f"position:exists:{deal_id}")
            continue
        row_epic = str(row.get("epic") or "").strip()
        if want and row_epic and row_epic != want:
            skipped.append(f"position:epic_mismatch:{deal_id}")
            continue
        side = row.get("side")
        if not isinstance(side, Side):
            direction = str(side or "").upper()
            if direction not in ("BUY", "SELL"):
                skipped.append(f"position:bad_side:{deal_id}")
                continue
            side = Side.BUY if direction == "BUY" else Side.SELL
        try:
            size = float(row.get("size") or 0)
            level = float(row.get("level") or 0)
        except (TypeError, ValueError):
            skipped.append(f"position:bad_numbers:{deal_id}")
            continue
        if size <= 0:
            skipped.append(f"position:zero_size:{deal_id}")
            continue

        if not ledger.positions:
            role = LegRole.PRIMARY
        else:
            first = next(iter(ledger.positions.values()))
            role = LegRole.PRIMARY if side == first.side else LegRole.HEDGE

        leg = ledger._open_leg(side, size, level, role)  # noqa: SLF001
        leg.deal_id = deal_id
        known_pos.add(deal_id)
        imported_positions.append(
            {"id": leg.id, "deal_id": deal_id, "side": side.value, "size": size, "level": level}
        )

    # --- working orders ---
    for raw in working_orders:
        parsed = _parse_ig_working_order_row(
            raw if isinstance(raw, dict) else {}, epic=want
        )
        if parsed is None:
            # Maybe already normalized by list helper
            if isinstance(raw, dict) and raw.get("deal_id"):
                parsed = raw
            else:
                skipped.append("order:unparseable")
                continue
        deal_id = str(parsed.get("deal_id") or "").strip()
        if not deal_id:
            skipped.append("order:missing_deal_id")
            continue
        if deal_id in known_wo:
            skipped.append(f"order:exists:{deal_id}")
            continue

        side = parsed.get("side")
        if not isinstance(side, Side):
            skipped.append(f"order:bad_side:{deal_id}")
            continue
        otype = parsed.get("type")
        if not isinstance(otype, OrderType):
            otype = OrderType.STOP if "STOP" in str(otype).upper() else OrderType.LIMIT
        level = float(parsed.get("level") or 0)
        size = float(parsed.get("size") or 0)
        if level <= 0 or size <= 0:
            skipped.append(f"order:bad_numbers:{deal_id}")
            continue

        purpose = OrderPurpose.ENTRY
        position_id: str | None = None
        # Heuristic: opposite an open leg → TP (limit) or hedge_cover (stop).
        for leg in ledger.positions.values():
            if leg.side != side:
                if otype == OrderType.LIMIT:
                    purpose = OrderPurpose.TP
                    position_id = leg.id
                else:
                    purpose = OrderPurpose.HEDGE_COVER
                    position_id = leg.id
                break

        order = WorkingOrder(
            id="",
            type=otype,
            side=side,
            level=level,
            size=size,
            purpose=purpose,
            position_id=position_id,
            deal_id=deal_id,
        )
        placed = ledger.place_order(order)
        known_wo.add(deal_id)
        order_book[placed.id] = deal_id
        imported_orders.append(
            {
                "id": placed.id,
                "deal_id": deal_id,
                "purpose": purpose.value,
                "side": side.value,
                "level": level,
            }
        )

    if not imported_positions and not imported_orders and not warnings:
        warnings.append("nothing_new")

    return {
        "imported_positions": imported_positions,
        "imported_orders": imported_orders,
        "skipped": skipped,
        "warnings": warnings,
        "order_book": order_book,
    }


def adopt_ig_book(
    session: Session,
    settings: Settings,
    slug: str,
) -> dict[str, Any]:
    """
    Fetch IG positions + working orders and merge into the local live ledger.

    Additive only (matched by deal_id). Updates the primary connector order book
    so the mirror neither cancels nor re-pushes imported rows.
    """
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository

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
    if live_cfg["mode"] == "off":
        return {
            "ok": False,
            "message": "Bot is Off — switch to Paper or Live first.",
            "error": "mode_off",
        }

    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    connectors, _warnings, conn_err = _resolve_selected_ig_connectors(
        conn_svc, tenant.id, live_cfg
    )
    if conn_err or not connectors:
        return {
            "ok": False,
            "message": "No selected active IG connector.",
            "error": conn_err or "no_connector",
        }

    primary_id, primary_cfg = connectors[0]
    integ_cfg = dict(integ.config or {})
    gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
    cfg = _build_cac40_config(
        live_cfg=live_cfg,
        integ_cfg=integ_cfg,
        primary_ig=primary_cfg,
        tenant_slug=slug,
        gemini_model=gemini_model,
    )

    connector = IgConnector(ig_config_from_connector(primary_cfg), dry_run=True)
    try:
        connector.login()
        if not connector.authenticated:
            return {
                "ok": False,
                "message": "IG login failed — check connector credentials.",
                "error": "login_failed",
            }
        # Force session for GETs even though dry_run (login sets _cst).
        positions = connector.list_open_positions(epic=cfg.epic)
        raw_orders = connector.list_working_orders()
    except Exception as exc:
        logger.exception("adopt_ig_book fetch failed for %s", slug)
        return {
            "ok": False,
            "message": f"IG fetch failed: {exc}",
            "error": "fetch_failed",
        }
    finally:
        try:
            connector.close()
        except Exception:
            pass

    with _LOCK:
        sched = _SCHEDULERS.get(slug)
        if sched is not None:
            ledger = sched.ig.ledger
            persist_via_sched = True
        else:
            state = _read_json(live_state_path(settings, slug), default=None)
            ledger = HedgeLedger.from_state_dict(
                cfg, state if isinstance(state, dict) else None
            )
            persist_via_sched = False

        result = adopt_ig_snapshot_into_ledger(
            ledger,
            positions=positions,
            working_orders=raw_orders,
            epic=cfg.epic,
        )

        # Merge deal ids into every selected connector order book so the mirror
        # neither cancels nor re-pushes adopted rows on secondary accounts.
        books_dir = live_dir(settings, slug) / "order_books"
        books_dir.mkdir(parents=True, exist_ok=True)
        connector_ids = [cid for cid, _ in connectors] or [primary_id]
        for cid in connector_ids:
            book_path = books_dir / f"orders_{cid}.json"
            book: dict[str, str] = {}
            if book_path.exists():
                try:
                    raw_book = json.loads(book_path.read_text(encoding="utf-8"))
                    if isinstance(raw_book, dict):
                        book = {str(k): str(v) for k, v in raw_book.items() if k and v}
                except Exception:
                    book = {}
            for oid, deal_id in (result.get("order_book") or {}).items():
                book[str(oid)] = str(deal_id)
            book_path.write_text(json.dumps(book, indent=2), encoding="utf-8")

        _write_json(live_state_path(settings, slug), ledger.to_state_dict())
        if persist_via_sched and sched is not None:
            sched.request_position_reconcile()

    n_pos = len(result.get("imported_positions") or [])
    n_ord = len(result.get("imported_orders") or [])
    msg = f"Synced from IG: +{n_pos} position(s), +{n_ord} order(s)."
    if n_pos == 0 and n_ord == 0:
        msg = "Sync from IG: nothing new to import (already in sync or empty)."
    return {
        "ok": True,
        "message": msg,
        **result,
    }


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
            _persist_llm_schedule(sched, settings, slug)
            _write_json(live_state_path(settings, slug), sched.ig.ledger.to_state_dict())
            _append_decision(settings, slug, payload)
            slot = _remember_cycle_slot(slug)
            pnl = sched.ig.ledger.pnl_payload()
            feed_status = _status_from_cycle_payload(payload, base_warnings=warnings)
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
                    **feed_status,
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
                _persist_llm_schedule(sched, settings, slug)
                _write_json(live_state_path(settings, slug), sched.ig.ledger.to_state_dict())
                _append_decision(settings, slug, payload)
                _remember_cycle_slot(slug, slot)
                feed_status = _status_from_cycle_payload(payload, base_warnings=warnings)
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
                        **feed_status,
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
