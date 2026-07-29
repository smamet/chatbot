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
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.application.tenant_service import TenantService
from chatbot.application.trader_backtest_service import default_ohlc_path
from chatbot.trader.config import TraderConfig, public_config_snapshot
from chatbot.trader.hedge_ledger import HedgeLedger
from chatbot.trader.ig_connector import IgConnector
from chatbot.trader.ig_ohlc import ig_config_from_connector
from chatbot.trader.live_ohlc_feed import prepare_live_ohlc_feed
from chatbot.trader.scheduler import LiveScheduler
from chatbot.config.settings import Settings

logger = logging.getLogger(__name__)

LIVE_MODES = frozenset({"off", "live"})
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
    """Live journal under data/trader/{slug}/live (migrates data/cac40/... once)."""
    from chatbot.application.trader_backtest_service import trader_root

    path = trader_root(settings, slug) / "live"
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


def sync_log_path(settings: Settings, slug: str) -> Path:
    return live_dir(settings, slug) / "sync_log.json"


SYNC_LOG_MAX = 200


def read_sync_log(settings: Settings, slug: str, *, limit: int = 100) -> list[dict[str, Any]]:
    raw = _read_json(sync_log_path(settings, slug), default=[]) or []
    if not isinstance(raw, list):
        return []
    rows = [r for r in raw if isinstance(r, dict)]
    return rows[: max(1, int(limit))]


def append_sync_log(settings: Settings, slug: str, entry: dict[str, Any]) -> None:
    path = sync_log_path(settings, slug)
    existing = _read_json(path, default=[]) or []
    if not isinstance(existing, list):
        existing = []
    row = dict(entry)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    rows = [row, *[r for r in existing if isinstance(r, dict)]]
    _write_json(path, rows[:SYNC_LOG_MAX])


def clear_sync_log(settings: Settings, slug: str) -> None:
    path = sync_log_path(settings, slug)
    if path.exists():
        path.unlink()


def _sync_changes_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract desync deltas from a cycle payload; None when nothing changed.

    Bare ``repair={mode: replace_open}`` after a matching wipe+rebuild is ignored —
    only true open-book deltas / unresolved desync are logged.
    """
    wo = payload.get("working_order_sync") if isinstance(payload, dict) else None
    rec = payload.get("reconcile") if isinstance(payload, dict) else None
    if not isinstance(wo, dict):
        wo = {}
    if not isinstance(rec, dict):
        rec = {}
    dropped = list(rec.get("dropped_orders") or wo.get("dropped") or [])
    imported_orders = list(rec.get("imported") or wo.get("imported") or [])
    opened = list(rec.get("opened") or [])
    closed = list(rec.get("closed") or [])
    quarantined = list(rec.get("quarantined") or [])
    repair = rec.get("repair") if isinstance(rec.get("repair"), dict) else None
    if not quarantined and repair:
        quarantined = list(repair.get("quarantined") or [])
    warnings = list(wo.get("warnings") or []) + list(rec.get("warnings") or [])
    unresolved = bool(rec.get("desync")) and not bool(rec.get("repaired"))
    has_deltas = bool(
        dropped or imported_orders or opened or closed or quarantined
    )
    if not has_deltas and not unresolved:
        return None
    return {
        "dropped": dropped,
        "imported_orders": imported_orders,
        "opened": opened,
        "closed": closed,
        "quarantined": quarantined,
        "repair": repair,
        "warnings": warnings,
        "desync": True,
        "changed": True,
    }


def append_sync_log_from_payload(
    settings: Settings,
    slug: str,
    payload: dict[str, Any],
    *,
    source: str = "cycle",
) -> bool:
    """Append a sync-log entry when the cycle changed the book. Returns True if logged."""
    changes = _sync_changes_from_payload(payload)
    if changes is None:
        return False
    append_sync_log(
        settings,
        slug,
        {
            "ts": payload.get("ts") or datetime.now(timezone.utc).isoformat(),
            "source": source,
            "cycle_id": payload.get("cycle_dir") or None,
            **changes,
        },
    )
    return True


def live_journal_dir(settings: Settings, slug: str) -> Path:
    path = live_dir(settings, slug) / "journal"
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_worker_status_path(settings: Settings) -> Path:
    return Path(settings.data_root) / "trader" / "live_worker_status.json"


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
    cfg = TraderConfig().public_snapshot()
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


def normalize_live_mode(mode: Any) -> str:
    """Map legacy ``paper`` / unknown values to ``off``; keep ``off``|``live``."""
    value = str(mode or "off").strip().lower()
    if value == "paper":
        return "off"
    if value in LIVE_MODES:
        return value
    return "off"


def load_live_config(settings: Settings, slug: str) -> dict[str, Any]:
    raw = _read_json(live_config_path(settings, slug), default=None)
    base = default_live_config()
    if not isinstance(raw, dict):
        return base
    raw_mode = str(raw.get("mode") or "off").strip().lower()
    mode = normalize_live_mode(raw_mode)
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
    cfg = {"mode": mode, "ig_connector_ids": ids, "strategy": strategy}
    # Self-heal live_config.json when legacy paper/unknown mode is on disk.
    if raw_mode != mode:
        _write_json(live_config_path(settings, slug), cfg)
    return cfg


def resolve_trader_trading_banner(
    session: Session,
    settings: Settings,
    *,
    tenant_id: int,
    slug: str,
    allowed_integrations: list[str] | tuple[str, ...] | None = None,
    bot_type: str | None = None,
) -> dict[str, Any] | None:
    """
    Topbar indicator when this bot is a trader.

    Returns None for assistant bots; otherwise
    ``{"active": True, "mode": "off"|"live", "slug": ..., "stream": "off"|"ok"|"stale"|"down"}``.
    """
    del session, allowed_integrations  # signature kept for call-site compatibility
    from chatbot.domain.models.tenant import BotType
    from chatbot.application.trader_stream_service import (
        read_stream_status,
        read_stream_worker_status,
        stream_is_healthy,
    )
    from chatbot.trader.market_calendar import session_snapshot

    if str(bot_type or "").strip().lower() != BotType.TRADER.value:
        return None
    mode = normalize_live_mode(load_live_config(settings, slug)["mode"])
    stream = "off"
    if mode == "live":
        stream_status = read_stream_status(settings, slug)
        stream_worker = read_stream_worker_status(settings)
        stream_worker_down = False
        try:
            hb = stream_worker.get("last_heartbeat_at") or stream_worker.get("finished_at")
            if hb:
                finished = datetime.fromisoformat(str(hb).replace("Z", "+00:00"))
                age = (
                    datetime.now(timezone.utc) - finished.astimezone(timezone.utc)
                ).total_seconds()
                loop = max(5.0, float(getattr(settings, "trader_stream_loop_seconds", 5) or 5))
                stream_worker_down = age > loop * 3
            else:
                stream_worker_down = True
        except Exception:
            stream_worker_down = True
        dealing_open = True
        try:
            dealing_open = bool(session_snapshot().get("dealing_open"))
        except Exception:
            pass
        if stream_worker_down:
            stream = "down"
        elif stream_status.get("stale") or not stream_status.get("connected"):
            stream = "stale"
        elif stream_is_healthy(stream_status, dealing_open=dealing_open):
            stream = "ok"
        else:
            stream = "stale"
    return {"active": True, "mode": mode, "slug": slug, "stream": stream}


def save_live_config(settings: Settings, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = load_live_config(settings, slug)
    mode = normalize_live_mode(payload.get("mode") or current["mode"])
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
    # Normalize bools / numbers via TraderConfig
    cfg = TraderConfig.from_dict(strategy)
    every_n, every_unit, every_bars = TraderConfig.llm_rate_from_form(
        every_n=int(cfg.llm_every_n or 6), unit=str(cfg.llm_every_unit or "1h")
    )
    strategy = {
        "max_open_positions": int(cfg.max_open_positions),
        "order_size": float(cfg.order_size),
        "spread_points": float(cfg.spread_points),
        "prevent_loss_exits": bool(cfg.prevent_loss_exits),
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
    raw_mode = str(mode or "").strip().lower()
    if raw_mode == "paper" or not raw_mode:
        mode = "off"
    elif raw_mode in LIVE_MODES:
        mode = raw_mode
    else:
        raise ValueError(f"Invalid mode {raw_mode!r}; expected off|live")
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
    # Force GET /positions on the next cycle after arming live.
    if mode == "live":
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


def _build_trader_config(
    *,
    live_cfg: dict[str, Any],
    integ_cfg: dict[str, Any],
    primary_ig: dict[str, Any],
    tenant_slug: str,
    gemini_model: str,
    system_prompt: str = "",
    market_profile: str = "cac40",
) -> TraderConfig:
    from chatbot.trader.profiles import get_profile

    strategy = dict(live_cfg.get("strategy") or {})
    ig_base = ig_config_from_connector(primary_ig)
    profile = get_profile(str(integ_cfg.get("market_profile") or market_profile))
    merged = {
        **strategy,
        "symbol": str(integ_cfg.get("symbol") or profile.default_symbol),
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
        "system_prompt": system_prompt or "",
        "market_profile": profile.id,
        "calendar_id": profile.calendar_id,
    }
    # Profile default unless strategy explicitly overrides.
    if "hedge_beyond_entry_points" not in strategy:
        merged["hedge_beyond_entry_points"] = float(profile.hedge_beyond_entry_points)
    if integ_cfg.get("max_open_positions") not in (None, "") and "max_open_positions" not in strategy:
        try:
            merged["max_open_positions"] = int(integ_cfg["max_open_positions"])
        except (TypeError, ValueError):
            pass
    cfg = TraderConfig.from_dict(merged)
    cfg.llm_mode = "live"
    cfg.llm_every_bars = cfg.resolve_llm_every_bars()
    return cfg


def _resolved_calendar_id(cfg: TraderConfig) -> str:
    if str(cfg.calendar_id or "").strip():
        return str(cfg.calendar_id).strip()
    from chatbot.trader.profiles import get_profile

    return get_profile(cfg.market_profile).calendar_id


def _config_hash(
    *,
    mode: str,
    connector_ids: list[int],
    cfg: TraderConfig,
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
    """Clear live journal/state. Blocked when mode is live."""
    cfg = load_live_config(settings, slug)
    current = mode or cfg["mode"]
    if current == "live":
        raise ValueError("Cannot clear history while Live mode is armed.")
    state = live_state_path(settings, slug)
    decisions = live_decisions_path(settings, slug)
    sync_log = sync_log_path(settings, slug)
    if state.exists():
        state.unlink()
    if decisions.exists():
        decisions.unlink()
    if sync_log.exists():
        sync_log.unlink()
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
                "url": f"/dashboard/bots/{slug}/trader/live/charts/{cycle}/{name}",
            }
        )
    return charts


def _decision_row_from_entry(
    entry: dict[str, Any], *, slug: str, journal_root: Path
) -> dict[str, Any]:
    """Map a cycle.json / decisions_log entry into run.html decision shape."""
    from chatbot.application.trader_cycle_ops_log import (
        build_cycle_ops_log,
        ops_log_line_count,
    )

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
    row = {
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
    ops_log = build_cycle_ops_log(row)
    row["ops_log"] = ops_log
    row["ops_log_line_count"] = ops_log_line_count(ops_log)
    return row


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


def _load_market_closed_groups(journal: Path) -> list[dict[str, Any]]:
    """One UI row per closed stretch (jsonl heartbeats under journal/market_closed/)."""
    root = journal / "market_closed"
    if not root.is_dir():
        return []
    groups: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        heartbeats: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("ts"):
                    heartbeats.append(row)
        except OSError:
            continue
        if not heartbeats:
            continue
        heartbeats.sort(key=lambda h: str(h.get("ts") or ""))
        first_ts = str(heartbeats[0].get("ts") or "")
        last_ts = str(heartbeats[-1].get("ts") or "")
        next_open = heartbeats[-1].get("next_open") or heartbeats[0].get("next_open")
        close_id = path.stem
        groups.append(
            {
                "kind": "market_closed_group",
                "ts": last_ts or first_ts,
                "from_ts": first_ts,
                "to_ts": last_ts,
                "next_open": next_open,
                "close_id": close_id,
                "heartbeat_count": len(heartbeats),
                "heartbeats": heartbeats,
                "cycle_dir": f"market_closed-{close_id}",
                "skipped": True,
                "skip_reason": "market_closed",
                "bias": None,
                "support": None,
                "resistance": None,
                "actions": [],
                "executed": [],
                "rejected": [],
                "charts": [],
                "chart_files": [],
                "decision": None,
                "book": {"positions": 0, "working_orders": 0},
                "pnl": {},
                "ops_log": [],
                "ops_log_line_count": 0,
            }
        )
    return groups


def merge_decisions_with_market_closed(
    decisions: list[dict[str, Any]],
    closed_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Interleave normal decision cards with market-closed groups (newest first)."""
    merged = list(decisions) + list(closed_groups)
    merged.sort(key=lambda e: str(e.get("ts") or ""), reverse=True)
    return merged


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
    closed_groups = _load_market_closed_groups(journal)
    return merge_decisions_with_market_closed(out, closed_groups)


def _as_dict_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [x for x in raw.values() if isinstance(x, dict)]
    return []


def read_live_book(settings: Settings, slug: str) -> dict[str, Any]:
    """Open book from local ledger state.json (positions + working orders)."""
    state_raw = _read_json(live_state_path(settings, slug), default={}) or {}
    if not isinstance(state_raw, dict):
        state_raw = {}
    positions = _as_dict_list(state_raw.get("positions"))
    working = _as_dict_list(state_raw.get("working_orders"))
    status = read_live_status(settings, slug)
    as_of = status.get("last_cycle_at") or status.get("finished_at")
    return {
        "positions": positions,
        "working_orders": working,
        "phase": state_raw.get("phase") or "Flat",
        "last_price": state_raw.get("last_price"),
        "as_of": as_of,
        "groups": group_open_book(positions, working),
    }


def build_live_panel_snapshot(
    settings: Settings, slug: str, *, cycle_limit: int = 3
) -> dict[str, Any]:
    """
    Lightweight payload for Trading → Live auto-refresh.

    ``fingerprint`` changes when open-book groups or the latest cycles change so
    the browser can skip DOM swaps when nothing moved.
    """
    book = read_live_book(settings, slug)
    cycles = list_live_cycles(settings, slug, limit=cycle_limit)
    live_cfg = load_live_config(settings, slug)
    status = read_live_status(settings, slug)
    fp_src = {
        "book_as_of": book.get("as_of"),
        "phase": book.get("phase"),
        "last_price": book.get("last_price"),
        "groups": book.get("groups"),
        "cycles": [
            {
                "cycle_id": c.get("cycle_id"),
                "ts": c.get("ts"),
                "skipped": c.get("skipped"),
                "bias": c.get("bias"),
                "has_charts": c.get("has_charts"),
                "executed_count": c.get("executed_count"),
                "rejected_count": c.get("rejected_count"),
            }
            for c in cycles
            if isinstance(c, dict)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(fp_src, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "mode": live_cfg.get("mode") or "off",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "book_as_of": book.get("as_of"),
        "last_cycle_at": status.get("last_cycle_at"),
        "fingerprint": digest,
        "book": book,
        "cycles": cycles,
    }


def _book_position_row(pos: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_kind": "position",
        "id": str(pos.get("id") or ""),
        "side": pos.get("side") or "—",
        "size": pos.get("size"),
        "level": pos.get("entry"),
        "purpose": pos.get("role") or "primary",
        "order_type": None,
        "link": None,
        "deal_id": str(pos.get("deal_id") or ""),
        "upl": pos.get("upl"),
    }


def _book_order_row(order: dict[str, Any]) -> dict[str, Any]:
    purpose = str(order.get("purpose") or "")
    link = order.get("position_id") or order.get("parent_order_id")
    return {
        "row_kind": "order",
        "id": str(order.get("id") or ""),
        "side": order.get("side") or "—",
        "size": order.get("size"),
        "level": order.get("level"),
        "purpose": purpose or "—",
        "order_type": order.get("type"),
        "link": str(link) if link else None,
        "deal_id": str(order.get("deal_id") or ""),
        "upl": None,
    }


def group_open_book(
    positions: list[dict[str, Any]],
    working_orders: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Group open legs with linked WOs (position_id) and pending entries with
    bracket children (parent_order_id). Leftover WOs become orphan groups.
    """
    orders = [o for o in working_orders if isinstance(o, dict)]
    used: set[str] = set()
    groups: list[dict[str, Any]] = []

    by_position: dict[str, list[dict[str, Any]]] = {}
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for order in orders:
        oid = str(order.get("id") or "")
        if not oid:
            continue
        pid = order.get("position_id")
        if pid:
            by_position.setdefault(str(pid), []).append(order)
        parent = order.get("parent_order_id")
        if parent:
            by_parent.setdefault(str(parent), []).append(order)

    for pos in positions:
        if not isinstance(pos, dict):
            continue
        pid = str(pos.get("id") or "")
        children_raw = by_position.get(pid, []) if pid else []
        children: list[dict[str, Any]] = []
        for order in children_raw:
            oid = str(order.get("id") or "")
            if oid and oid not in used:
                used.add(oid)
                children.append(_book_order_row(order))
        groups.append(
            {
                "kind": "position",
                "parent": _book_position_row(pos),
                "children": children,
            }
        )

    for order in orders:
        oid = str(order.get("id") or "")
        if not oid or oid in used:
            continue
        purpose = str(order.get("purpose") or "").lower()
        if purpose != "entry":
            continue
        used.add(oid)
        children: list[dict[str, Any]] = []
        for child in by_parent.get(oid, []):
            cid = str(child.get("id") or "")
            if cid and cid not in used:
                used.add(cid)
                children.append(_book_order_row(child))
        groups.append(
            {
                "kind": "entry",
                "parent": _book_order_row(order),
                "children": children,
            }
        )

    for order in orders:
        oid = str(order.get("id") or "")
        if not oid or oid in used:
            continue
        used.add(oid)
        groups.append(
            {
                "kind": "orphan",
                "parent": _book_order_row(order),
                "children": [],
            }
        )

    return groups


def get_live_report(settings: Settings, slug: str) -> dict[str, Any]:
    """Build a run-like report payload for the live results page."""
    live_cfg = load_live_config(settings, slug)
    status = read_live_status(settings, slug)
    state_raw = _read_json(live_state_path(settings, slug), default={}) or {}
    if not isinstance(state_raw, dict):
        state_raw = {}
    root = live_dir(settings, slug)
    decisions = _load_live_decision_entries(settings, slug)

    book = read_live_book(settings, slug)
    positions = book["positions"]
    working = book["working_orders"]
    closed_raw = state_raw.get("closed_trades") or []
    if not isinstance(closed_raw, list):
        closed_raw = []
    closed_all = [t for t in closed_raw if isinstance(t, dict)]
    closed = [t for t in closed_all if not t.get("phantom")]
    phantom_closed = [t for t in closed_all if t.get("phantom")]
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
                    "deal_id": t.get("deal_id"),
                    "phantom": bool(t.get("phantom")),
                    "ig_confirmed": bool(t.get("ig_confirmed")),
                }
                for t in closed_all
            ],
            "phantom_closed_trades": [
                {
                    "id": t.get("id"),
                    "deal_id": t.get("deal_id"),
                    "pnl": t.get("realized_pnl"),
                    "phantom": True,
                    "ig_confirmed": bool(t.get("ig_confirmed")),
                }
                for t in phantom_closed
            ],
            "summary": {
                "cycles": status.get("cycles") or len(decisions),
                "open_legs": len(positions),
                "realized": realized,
                "net_upl": net_upl,
                "phantom_closes": len(phantom_closed),
            },
            "book": book,
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
        from chatbot.application.trader_stream_service import (
            read_stream_quote,
            read_stream_status,
            stream_is_healthy,
        )
        from chatbot.trader.market_calendar import session_snapshot

        stream_status = read_stream_status(settings, slug)
        quote = read_stream_quote(settings, slug)
        dealing_open = True
        try:
            dealing_open = bool(session_snapshot().get("dealing_open"))
        except Exception:
            pass
        healthy = stream_is_healthy(stream_status, dealing_open=dealing_open)
        stale = bool(stream_status.get("stale")) if stream_status else False
        # Only treat missing stream as non-stale when worker never wrote status
        # (stream worker not deployed) — then fall back to REST top-up.
        stream_configured = bool(stream_status)
        mid = None
        try:
            mid = float(quote.get("mid") or 0) or None
        except (TypeError, ValueError):
            mid = None
        return prepare_live_ohlc_feed(
            path,
            config=sched.config,
            connector=sched.ig,
            top_up=True,
            stream_healthy=healthy if stream_configured else None,
            stream_stale=stale if stream_configured else False,
            stream_mid=mid,
            stream_error=str(stream_status.get("stale_reason") or stream_status.get("error") or "")
            or None,
        )

    sched.ohlc_provider = _provider
    # Allow live cycle to skip redundant REST book sync when stream just reconciled.
    sched.stream_status_path = stream_status_path_for_slug(settings, slug)


def stream_status_path_for_slug(settings: Settings, slug: str):
    from chatbot.application.trader_stream_service import stream_status_path

    return stream_status_path(settings, slug)


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
    cfg: TraderConfig,
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


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


def _attached_levels_from_wo_payload(
    data: dict[str, Any], *, side: Any, level: float
) -> tuple[float | None, float | None]:
    """
    Resolve attached TP/SL on an IG working order.

    Prefers absolute ``limitLevel`` / ``stopLevel``. Falls back to
    ``limitDistance`` / ``stopDistance`` (IG FX often returns distance only).
    """
    from chatbot.trader.models import Side

    limit_level = _optional_positive_float(
        data.get("limitLevel") if data.get("limitLevel") is not None else data.get("limit_level")
    )
    stop_level = _optional_positive_float(
        data.get("stopLevel") if data.get("stopLevel") is not None else data.get("stop_level")
    )
    side_s = side if isinstance(side, Side) else None
    if side_s is None:
        token = str(getattr(side, "value", side) or "").upper()
        side_s = Side.BUY if token == "BUY" else Side.SELL if token == "SELL" else None
    if side_s is None or level <= 0:
        return limit_level, stop_level

    # FX mid < 50 → 0.0001/pt; indices → 1.0/pt (same heuristic as WO TP attach).
    pip = 0.0001 if level < 50 else 1.0
    if limit_level is None:
        dist = _optional_positive_float(
            data.get("limitDistance")
            if data.get("limitDistance") is not None
            else data.get("limit_distance")
        )
        if dist is not None:
            limit_level = (
                level + dist * pip if side_s == Side.BUY else level - dist * pip
            )
    if stop_level is None:
        dist = _optional_positive_float(
            data.get("stopDistance")
            if data.get("stopDistance") is not None
            else data.get("stop_distance")
        )
        if dist is not None:
            stop_level = (
                level - dist * pip if side_s == Side.BUY else level + dist * pip
            )
    return limit_level, stop_level


def _parse_ig_working_order_row(
    row: dict[str, Any], *, epic: str
) -> dict[str, Any] | None:
    """Normalize an IG working-order payload into side/type/level/size/deal_id."""
    from chatbot.trader.models import OrderType, Side

    data = row.get("workingOrderData") if isinstance(row.get("workingOrderData"), dict) else row
    if not isinstance(data, dict):
        return None
    row_epic = str(data.get("epic") or "").strip()
    if epic and row_epic and row_epic != epic:
        return None
    deal_id = str(data.get("dealId") or data.get("deal_id") or "").strip()
    if not deal_id:
        return None
    direction = str(data.get("direction") or "").upper()
    if direction not in ("BUY", "SELL"):
        side_raw = data.get("side")
        direction = str(getattr(side_raw, "value", side_raw) or "").upper()
    if direction not in ("BUY", "SELL"):
        return None
    side = Side.BUY if direction == "BUY" else Side.SELL
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
    limit_level, stop_level = _attached_levels_from_wo_payload(
        data, side=side, level=level
    )
    return {
        "deal_id": deal_id,
        "side": side,
        "type": otype,
        "level": level,
        "size": size,
        "epic": row_epic,
        "limit_level": limit_level,
        "stop_level": stop_level,
    }


def _import_attached_protection(
    ledger: HedgeLedger,
    *,
    parent_deal_id: str,
    close_side: Any,
    size: float,
    limit_level: float | None,
    stop_level: float | None,
    position_id: str | None = None,
    parent_order_id: str | None = None,
    known_wo: set[str],
    order_book: dict[str, str],
    imported_orders: list[dict[str, Any]],
) -> None:
    """Model IG attached limit/stop as local TP / hedge_cover WOs (sentinel deal ids)."""
    from chatbot.trader.models import (
        OrderPurpose,
        OrderType,
        WorkingOrder,
        attached_deal_id,
    )

    for purpose, attached_lvl, otype in (
        (OrderPurpose.TP, limit_level, OrderType.LIMIT),
        (OrderPurpose.HEDGE_COVER, stop_level, OrderType.STOP),
    ):
        if attached_lvl is None:
            continue
        try:
            lvl = float(attached_lvl)
        except (TypeError, ValueError):
            continue
        if lvl <= 0:
            continue
        sentinel = attached_deal_id(parent_deal_id, purpose)
        if sentinel in known_wo:
            continue
        twin = None
        if position_id:
            twin = _find_attached_protection_twin(
                ledger,
                purpose=purpose,
                side=close_side,
                level=lvl,
                leg_id=position_id,
            )
        if twin is not None:
            old_did = (twin.deal_id or "").strip()
            if old_did and old_did in known_wo:
                known_wo.discard(old_did)
            twin.deal_id = sentinel
            twin.position_id = position_id
            twin.parent_order_id = parent_order_id
            twin.level = lvl
            twin.size = size
            known_wo.add(sentinel)
            order_book[twin.id] = sentinel
            imported_orders.append(
                {
                    "id": twin.id,
                    "deal_id": sentinel,
                    "purpose": purpose.value,
                    "side": getattr(close_side, "value", close_side),
                    "level": lvl,
                    "upgraded": True,
                }
            )
            continue
        placed = ledger.place_order(
            WorkingOrder(
                id="",
                type=otype,
                side=close_side,
                level=lvl,
                size=size,
                purpose=purpose,
                position_id=position_id,
                parent_order_id=parent_order_id,
                deal_id=sentinel,
            )
        )
        known_wo.add(sentinel)
        order_book[placed.id] = sentinel
        imported_orders.append(
            {
                "id": placed.id,
                "deal_id": sentinel,
                "purpose": purpose.value,
                "side": getattr(close_side, "value", close_side),
                "level": lvl,
            }
        )


def _find_attached_protection_twin(
    ledger: HedgeLedger,
    *,
    purpose: Any,
    side: Any,
    level: float,
    leg_id: str,
) -> Any | None:
    """
    Find a local WO that already models the same attached stop/limit protection.

    Matches empty deal_id, any ``attached:…`` sentinel, or a child of a vanished
    entry — same side/level, not bound to a different open leg.
    """
    from chatbot.trader.models import OrderPurpose, parse_attached_deal_id

    want_purpose = purpose if isinstance(purpose, OrderPurpose) else OrderPurpose(str(purpose))
    open_leg_ids = set(ledger.positions.keys())
    for order in ledger.working_orders.values():
        purpose_ok = order.purpose == want_purpose or (
            want_purpose == OrderPurpose.TP and order.purpose == OrderPurpose.CLOSE
        )
        if not purpose_ok:
            continue
        if order.side != side:
            continue
        if abs(float(order.level) - float(level)) > 1e-6:
            continue
        if (
            order.position_id
            and order.position_id in open_leg_ids
            and order.position_id != leg_id
        ):
            continue
        did = (order.deal_id or "").strip()
        parent_still_working = bool(
            order.parent_order_id and order.parent_order_id in ledger.working_orders
        )
        if not did or parse_attached_deal_id(did) is not None:
            return order
        if order.parent_order_id and not parent_still_working:
            return order
    return None


def _is_ig_book_deal_id(deal_id: str) -> bool:
    """True for real IG deal ids (not empty / attached: sentinels)."""
    did = (deal_id or "").strip()
    return bool(did) and not did.startswith("attached:")


def _side_token(side: Any) -> str:
    if hasattr(side, "value"):
        return str(side.value).upper()
    return str(side or "").upper()


def _type_token(otype: Any) -> str:
    if hasattr(otype, "value"):
        return str(otype.value).upper()
    raw = str(otype or "LIMIT").upper()
    return "STOP" if "STOP" in raw else "LIMIT"


def _qty(value: Any) -> float | None:
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return round(n, 6)


def _position_book_key(deal_id: str, side: Any, size: Any) -> tuple[str, str, float] | None:
    if not _is_ig_book_deal_id(deal_id):
        return None
    side_s = _side_token(side)
    if side_s not in ("BUY", "SELL"):
        return None
    qty = _qty(size)
    if qty is None:
        return None
    return (deal_id.strip(), side_s, qty)


def _order_book_key(
    deal_id: str, side: Any, size: Any, level: Any, otype: Any
) -> tuple[str, str, float, float, str] | None:
    if not _is_ig_book_deal_id(deal_id):
        return None
    side_s = _side_token(side)
    if side_s not in ("BUY", "SELL"):
        return None
    qty = _qty(size)
    lvl = _qty(level)
    if qty is None or lvl is None:
        return None
    return (deal_id.strip(), side_s, qty, lvl, _type_token(otype))


def _local_position_keys(ledger: HedgeLedger) -> set[tuple[str, str, float]]:
    keys: set[tuple[str, str, float]] = set()
    for leg in ledger.positions.values():
        key = _position_book_key(leg.deal_id or "", leg.side, leg.size)
        if key:
            keys.add(key)
    return keys


def _local_order_keys(
    ledger: HedgeLedger,
) -> set[tuple[str, str, float, float, str]]:
    keys: set[tuple[str, str, float, float, str]] = set()
    for order in ledger.working_orders.values():
        key = _order_book_key(
            order.deal_id or "", order.side, order.size, order.level, order.type
        )
        if key:
            keys.add(key)
    return keys


def _ig_position_keys(
    positions: list[dict[str, Any]],
) -> set[tuple[str, str, float]]:
    keys: set[tuple[str, str, float]] = set()
    for row in positions:
        if not isinstance(row, dict):
            continue
        deal_id = str(row.get("deal_id") or row.get("dealId") or "").strip()
        key = _position_book_key(deal_id, row.get("side") or row.get("direction"), row.get("size"))
        if key:
            keys.add(key)
    return keys


def _ig_order_keys(
    working_orders: list[Any], *, epic: str
) -> set[tuple[str, str, float, float, str]]:
    keys: set[tuple[str, str, float, float, str]] = set()
    for raw in working_orders or []:
        parsed: dict[str, Any] | None = None
        if isinstance(raw, dict):
            parsed = _parse_ig_working_order_row(raw, epic=epic)
            if parsed is None and raw.get("deal_id"):
                parsed = raw
        if not isinstance(parsed, dict):
            continue
        deal_id = str(parsed.get("deal_id") or parsed.get("dealId") or "").strip()
        key = _order_book_key(
            deal_id,
            parsed.get("side") or parsed.get("direction"),
            parsed.get("size") or parsed.get("orderSize"),
            parsed.get("level") or parsed.get("orderLevel"),
            parsed.get("type") or parsed.get("orderType"),
        )
        if key:
            keys.add(key)
    return keys


def _ig_position_delta_rows(
    positions: list[dict[str, Any]],
    *,
    wanted: set[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for row in positions:
        if not isinstance(row, dict):
            continue
        deal_id = str(row.get("deal_id") or row.get("dealId") or "").strip()
        key = _position_book_key(deal_id, row.get("side") or row.get("direction"), row.get("size"))
        if key is None or key not in wanted or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "deal_id": key[0],
                "side": key[1],
                "size": key[2],
                "level": row.get("level") or row.get("openLevel"),
            }
        )
    return out


def _ig_order_delta_rows(
    working_orders: list[Any],
    *,
    epic: str,
    wanted: set[tuple[str, str, float, float, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float, str]] = set()
    for raw in working_orders or []:
        parsed: dict[str, Any] | None = None
        if isinstance(raw, dict):
            parsed = _parse_ig_working_order_row(raw, epic=epic)
            if parsed is None and raw.get("deal_id"):
                parsed = raw
        if not isinstance(parsed, dict):
            continue
        deal_id = str(parsed.get("deal_id") or parsed.get("dealId") or "").strip()
        key = _order_book_key(
            deal_id,
            parsed.get("side") or parsed.get("direction"),
            parsed.get("size") or parsed.get("orderSize"),
            parsed.get("level") or parsed.get("orderLevel"),
            parsed.get("type") or parsed.get("orderType"),
        )
        if key is None or key not in wanted or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "deal_id": key[0],
                "side": key[1],
                "size": key[2],
                "level": key[3],
                "type": key[4],
            }
        )
    return out


def _local_order_delta_rows(
    ledger: HedgeLedger,
    *,
    wanted: set[tuple[str, str, float, float, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float, str]] = set()
    for order in ledger.working_orders.values():
        key = _order_book_key(
            order.deal_id or "", order.side, order.size, order.level, order.type
        )
        if key is None or key not in wanted or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "order_id": order.id,
                "deal_id": key[0],
                "side": key[1],
                "size": key[2],
                "level": key[3],
                "type": key[4],
                "purpose": getattr(order.purpose, "value", str(order.purpose or "")),
            }
        )
    return out


def sync_open_book_from_ig(
    ledger: HedgeLedger,
    *,
    positions: list[dict[str, Any]],
    working_orders: list[Any],
    epic: str = "",
    exit_price_for_leg: Any | None = None,
) -> dict[str, Any]:
    """
    Rebuild the open book from IG — same path for dashboard Apply and every live cycle.

    1. Close local legs whose IG ``deal_id`` vanished (session PnL / ``ig_confirmed``).
    2. ``replace_open`` adopt of positions + working orders (IG is sole open-book SoT).

    Returns true open-book deltas in ``opened`` / ``imported`` / ``dropped_orders`` /
    ``closed`` (pre-wipe vs IG). ``changed`` is True only when those deltas (or
    quarantine) are non-empty — wipe+rebuild still always runs.
    """
    want = (epic or "").strip()
    ig_deal_ids = {
        str(row.get("deal_id") or row.get("dealId") or "").strip()
        for row in positions
        if isinstance(row, dict)
        and _is_ig_book_deal_id(str(row.get("deal_id") or row.get("dealId") or ""))
    }
    ig_net = 0.0
    for row in positions:
        if not isinstance(row, dict):
            continue
        try:
            size = float(row.get("size") or 0)
        except (TypeError, ValueError):
            continue
        side_s = _side_token(row.get("side") or row.get("direction"))
        if side_s == "BUY":
            ig_net += size
        elif side_s == "SELL":
            ig_net -= size

    local_pos_keys = _local_position_keys(ledger)
    local_wo_keys = _local_order_keys(ledger)
    ig_pos_keys = _ig_position_keys(positions)
    ig_wo_keys = _ig_order_keys(working_orders, epic=want)
    opened_keys = ig_pos_keys - local_pos_keys
    imported_keys = ig_wo_keys - local_wo_keys
    dropped_wo_keys = local_wo_keys - ig_wo_keys
    opened_delta = _ig_position_delta_rows(positions, wanted=opened_keys)
    imported_delta = _ig_order_delta_rows(
        working_orders, epic=want, wanted=imported_keys
    )
    dropped_orders = _local_order_delta_rows(ledger, wanted=dropped_wo_keys)

    closed: list[dict[str, Any]] = []
    for leg in list(ledger.positions.values()):
        deal_id = (leg.deal_id or "").strip()
        if not deal_id or deal_id in ig_deal_ids:
            continue
        if callable(exit_price_for_leg):
            exit_px = float(exit_price_for_leg(leg))
        else:
            px = float(ledger.last_price or 0)
            exit_px = px if px > 0 else float(leg.entry)
        trade = ledger.close_position(leg.id, exit_px, ig_confirmed=True)
        closed.append(
            {
                "id": leg.id,
                "deal_id": deal_id,
                "exit": exit_px,
                "trade_id": trade.id if trade else None,
            }
        )

    adopt = adopt_ig_snapshot_into_ledger(
        ledger,
        positions=positions,
        working_orders=list(working_orders or []),
        epic=epic,
        mode="replace_open",
    )
    quarantined = list(adopt.get("quarantined") or [])
    changed = bool(
        closed or opened_delta or imported_delta or dropped_orders or quarantined
    )
    return {
        **adopt,
        "closed": closed,
        "opened": opened_delta,
        "imported": imported_delta,
        "dropped_orders": dropped_orders,
        "ig_net": ig_net,
        "local_net": float(ledger.net_size()),
        "desync": False,
        "changed": changed,
        "repaired": True,
        "repair": {
            "imported_positions": adopt.get("imported_positions"),
            "imported_orders": adopt.get("imported_orders"),
            "quarantined": quarantined,
            "mode": "replace_open",
            "changed": changed,
        },
    }


def adopt_ig_snapshot_into_ledger(
    ledger: HedgeLedger,
    *,
    positions: list[dict[str, Any]],
    working_orders: list[dict[str, Any]],
    epic: str = "",
    mode: str = "additive",
) -> dict[str, Any]:
    """
    Import IG positions/orders into a local ledger.

    ``mode``:
    - ``additive`` (default): idempotent by deal_id; never deletes local open legs.
    - ``replace_open``: clear open positions/WOs, quarantine phantom closes for
      reappearing dealIds, then import IG snapshot as the open book.

    Returns ``{imported_positions, imported_orders, skipped, warnings, order_book,
    quarantined, replaced}``.
    """
    from chatbot.trader.models import (
        LegRole,
        OrderPurpose,
        OrderType,
        Side,
        WorkingOrder,
    )

    want = (epic or "").strip()
    mode_l = (mode or "additive").strip().lower()
    replaced = False
    quarantined: list[str] = []

    ig_deal_ids = {
        str(row.get("deal_id") or "").strip()
        for row in positions
        if isinstance(row, dict) and str(row.get("deal_id") or "").strip()
    }

    if mode_l == "replace_open":
        quarantined = ledger.quarantine_phantom_closes(ig_deal_ids)
        ledger.positions.clear()
        ledger.working_orders.clear()
        replaced = True

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
            # Refresh size/entry from IG when already present.
            for leg in ledger.positions.values():
                if (leg.deal_id or "").strip() == deal_id:
                    try:
                        leg.size = float(row.get("size") or leg.size)
                        level = float(row.get("level") or 0)
                        if level > 0:
                            leg.entry = level
                    except (TypeError, ValueError):
                        pass
                    break
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

        leg = ledger._open_leg(side, size, level, role, deal_id=deal_id)  # noqa: SLF001
        known_pos.add(deal_id)
        imported_positions.append(
            {"id": leg.id, "deal_id": deal_id, "side": side.value, "size": size, "level": level}
        )

        # Model attached stop/limit from the position payload (not in /workingorders).
        close_side = Side.SELL if side == Side.BUY else Side.BUY
        _import_attached_protection(
            ledger,
            parent_deal_id=deal_id,
            close_side=close_side,
            size=size,
            limit_level=_optional_positive_float(row.get("limit_level")),
            stop_level=_optional_positive_float(row.get("stop_level")),
            position_id=leg.id,
            known_wo=known_wo,
            order_book=order_book,
            imported_orders=imported_orders,
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
        # IG /workingorders are standalone forceOpen orders (Orders tab).
        # True position TP lives on position.limitLevel → attached:* above.
        # Opposite-side LIMIT must stay ENTRY (new short/long), not fake TP.
        # Opposite-side STOP may be our forceOpen hedge_cover — nest for UX.
        if otype == OrderType.STOP:
            for leg in ledger.positions.values():
                if leg.side != side:
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

        # IG can attach TP/SL on the working-order ticket (limitLevel/stopLevel
        # or *Distance). Nest under the entry via parent_order_id — do not treat
        # separate opposite LIMITs as TP (those stay ENTRY above).
        if purpose == OrderPurpose.ENTRY:
            close_side = Side.SELL if side == Side.BUY else Side.BUY
            lim = _optional_positive_float(parsed.get("limit_level"))
            stp = _optional_positive_float(parsed.get("stop_level"))
            if lim is None and stp is None and isinstance(raw, dict):
                lim, stp = _attached_levels_from_wo_payload(
                    raw.get("workingOrderData")
                    if isinstance(raw.get("workingOrderData"), dict)
                    else raw,
                    side=side,
                    level=level,
                )
            _import_attached_protection(
                ledger,
                parent_deal_id=deal_id,
                close_side=close_side,
                size=size,
                limit_level=lim,
                stop_level=stp,
                parent_order_id=placed.id,
                known_wo=known_wo,
                order_book=order_book,
                imported_orders=imported_orders,
            )

    if not imported_positions and not imported_orders and not warnings and not replaced:
        warnings.append("nothing_new")

    ledger.infer_phase()
    if ledger.last_price:
        ledger.mark_to_market(ledger.last_price)

    return {
        "imported_positions": imported_positions,
        "imported_orders": imported_orders,
        "skipped": skipped,
        "warnings": warnings,
        "order_book": order_book,
        "quarantined": quarantined,
        "replaced": replaced,
        "mode": mode_l,
    }


def _fetch_ig_snapshot(
    session: Session,
    settings: Settings,
    slug: str,
) -> dict[str, Any]:
    """
    Login to primary IG connector and fetch open positions + working orders.

    Returns ``{ok, cfg, connectors, primary_id, positions, raw_orders, live_cfg}``
    or ``{ok: False, message, error}``.
    """
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository

    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    tenant = tenant_svc.get_by_slug(slug)
    if tenant is None:
        return {"ok": False, "message": f"Bot {slug!r} not found.", "error": "not_found"}

    if not tenant.is_trader:
        return {
            "ok": False,
            "message": "This bot is not a trader bot.",
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
    from chatbot.domain.trader_access import trader_settings_as_integration_dict

    integ_cfg = trader_settings_as_integration_dict(tenant)
    gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
    cfg = _build_trader_config(
        live_cfg=live_cfg,
        integ_cfg=integ_cfg,
        primary_ig=primary_cfg,
        tenant_slug=slug,
        gemini_model=gemini_model,
        system_prompt=str(tenant.prompt or ""),
        market_profile=str(getattr(tenant.config.trader, "market_profile", None) or "cac40"),
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
        positions = connector.list_open_positions(epic=cfg.epic)
        raw_orders = connector.list_working_orders()
    except Exception as exc:
        logger.exception("IG snapshot fetch failed for %s", slug)
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

    return {
        "ok": True,
        "cfg": cfg,
        "connectors": connectors,
        "primary_id": primary_id,
        "positions": positions,
        "raw_orders": raw_orders,
        "live_cfg": live_cfg,
    }


def _ig_deal_sets(
    positions: list[dict[str, Any]],
    raw_orders: list[Any],
    *,
    epic: str,
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    """Return (position_deal_ids, working_order_deal_ids incl. attached, pos_by_deal)."""
    pos_by_deal: dict[str, dict[str, Any]] = {}
    pos_ids: set[str] = set()
    wo_ids: set[str] = set()
    for row in positions:
        if not isinstance(row, dict):
            continue
        did = str(row.get("deal_id") or "").strip()
        if not did:
            continue
        pos_ids.add(did)
        pos_by_deal[did] = row
        for purpose, lvl_key in (("tp", "limit_level"), ("hedge_cover", "stop_level")):
            raw_lvl = row.get(lvl_key)
            if raw_lvl is None:
                continue
            try:
                if float(raw_lvl) > 0:
                    wo_ids.add(f"attached:{did}:{purpose}")
            except (TypeError, ValueError):
                continue
    for raw in raw_orders:
        parsed = _parse_ig_working_order_row(
            raw if isinstance(raw, dict) else {}, epic=epic
        )
        if parsed is None and isinstance(raw, dict) and raw.get("deal_id"):
            parsed = raw
        if not parsed:
            continue
        did = str(parsed.get("deal_id") or "").strip()
        if did:
            wo_ids.add(did)
            for purpose, lvl_key in (("tp", "limit_level"), ("hedge_cover", "stop_level")):
                raw_lvl = parsed.get(lvl_key)
                if raw_lvl is None:
                    continue
                try:
                    if float(raw_lvl) > 0:
                        wo_ids.add(f"attached:{did}:{purpose}")
                except (TypeError, ValueError):
                    continue
    return pos_ids, wo_ids, pos_by_deal


def _annotate_status(row: dict[str, Any], status: str) -> dict[str, Any]:
    out = dict(row)
    out["status"] = status
    return out


def preview_ig_book(
    session: Session,
    settings: Settings,
    slug: str,
) -> dict[str, Any]:
    """
    Diff local open book vs IG without mutating state.

    Runs ``adopt_ig_snapshot_into_ledger(..., mode=replace_open)`` on a throwaway
    ledger copy so the preview matches what Apply sync would do.
    """
    snap = _fetch_ig_snapshot(session, settings, slug)
    if not snap.get("ok"):
        return snap

    cfg = snap["cfg"]
    positions_ig: list[dict[str, Any]] = list(snap["positions"] or [])
    raw_orders: list[Any] = list(snap["raw_orders"] or [])
    live_cfg = snap["live_cfg"]
    ig_pos_ids, ig_wo_ids, ig_pos_by_deal = _ig_deal_sets(
        positions_ig, raw_orders, epic=cfg.epic
    )

    state_raw = _read_json(live_state_path(settings, slug), default={}) or {}
    if not isinstance(state_raw, dict):
        state_raw = {}
    local_positions = _as_dict_list(state_raw.get("positions"))
    local_orders = _as_dict_list(state_raw.get("working_orders"))
    closed_raw = state_raw.get("closed_trades") or []
    if not isinstance(closed_raw, list):
        closed_raw = []

    local_pos_deals = {
        str(p.get("deal_id") or "").strip()
        for p in local_positions
        if str(p.get("deal_id") or "").strip()
    }
    local_wo_deals = {
        str(o.get("deal_id") or "").strip()
        for o in local_orders
        if str(o.get("deal_id") or "").strip()
    }

    # Throwaway adopt — real state untouched.
    copy = HedgeLedger.from_state_dict(cfg, state_raw)
    adopt_result = adopt_ig_snapshot_into_ledger(
        copy,
        positions=positions_ig,
        working_orders=raw_orders,
        epic=cfg.epic,
        mode="replace_open",
    )

    annotated_positions: list[dict[str, Any]] = []
    for pos in local_positions:
        row = _book_position_row(pos)
        did = str(pos.get("deal_id") or "").strip()
        if did and did in ig_pos_ids:
            status = "in_sync"
        else:
            status = "remove"
        annotated_positions.append(_annotate_status(row, status))

    annotated_orders: list[dict[str, Any]] = []
    for order in local_orders:
        row = _book_order_row(order)
        did = str(order.get("deal_id") or "").strip()
        if did.startswith("attached:"):
            # Position- and WO-attached TP/SL are both recorded in ig_wo_ids as
            # attached:{parentDeal}:{purpose} when IG still has the level.
            status = "in_sync" if did in ig_wo_ids else "remove"
        elif did and did in ig_wo_ids:
            status = "in_sync"
        else:
            status = "remove"
        annotated_orders.append(_annotate_status(row, status))

    # New rows from adopt imports that were not already local.
    new_positions: list[dict[str, Any]] = []
    for imp in adopt_result.get("imported_positions") or []:
        if not isinstance(imp, dict):
            continue
        did = str(imp.get("deal_id") or "").strip()
        if did and did in local_pos_deals:
            continue
        new_positions.append(
            _annotate_status(
                {
                    "row_kind": "position",
                    "id": str(imp.get("id") or ""),
                    "side": imp.get("side") or "—",
                    "size": imp.get("size"),
                    "level": imp.get("level"),
                    "purpose": "primary",
                    "order_type": None,
                    "link": None,
                    "deal_id": did,
                    "upl": None,
                },
                "new",
            )
        )

    new_orders: list[dict[str, Any]] = []
    for imp in adopt_result.get("imported_orders") or []:
        if not isinstance(imp, dict):
            continue
        did = str(imp.get("deal_id") or "").strip()
        if did and did in local_wo_deals:
            continue
        new_orders.append(
            _annotate_status(
                {
                    "row_kind": "order",
                    "id": str(imp.get("id") or ""),
                    "side": imp.get("side") or "—",
                    "size": imp.get("size"),
                    "level": imp.get("level"),
                    "purpose": imp.get("purpose") or "—",
                    "order_type": None,
                    "link": None,
                    "deal_id": did,
                    "upl": None,
                },
                "new",
            )
        )

    # Build groups: keep local grouping, then orphan "new" rows.
    local_groups = group_open_book(local_positions, local_orders)
    status_by_id = {
        r["id"]: r["status"]
        for r in (*annotated_positions, *annotated_orders)
        if r.get("id")
    }
    groups: list[dict[str, Any]] = []
    for g in local_groups:
        parent = dict(g["parent"])
        parent["status"] = status_by_id.get(parent.get("id") or "", "remove")
        children = []
        for child in g.get("children") or []:
            c = dict(child)
            c["status"] = status_by_id.get(c.get("id") or "", "remove")
            children.append(c)
        groups.append({"kind": g["kind"], "parent": parent, "children": children})

    for row in new_positions:
        groups.append({"kind": "position", "parent": row, "children": []})
    for row in new_orders:
        groups.append({"kind": "orphan", "parent": row, "children": []})

    closed_rows: list[dict[str, Any]] = []
    for t in closed_raw:
        if not isinstance(t, dict):
            continue
        did = str(t.get("deal_id") or "").strip()
        phantom = bool(t.get("phantom"))
        ig_confirmed = bool(t.get("ig_confirmed"))
        if not did:
            status = "paper"
        elif did in ig_pos_ids:
            status = "reopened"
        elif phantom:
            status = "reopened"
        elif ig_confirmed:
            status = "confirmed"
        else:
            status = "verified_closed"
        closed_rows.append(
            {
                "id": t.get("id"),
                "side": t.get("side"),
                "role": t.get("role"),
                "entry": t.get("entry"),
                "exit": t.get("exit"),
                "pnl": t.get("realized_pnl"),
                "deal_id": did,
                "phantom": phantom,
                "ig_confirmed": ig_confirmed,
                "status": status,
            }
        )

    n_remove = sum(
        1
        for g in groups
        for r in (g["parent"], *(g.get("children") or []))
        if r.get("status") == "remove"
    )
    n_new = sum(
        1
        for g in groups
        for r in (g["parent"], *(g.get("children") or []))
        if r.get("status") == "new"
    )
    n_sync = sum(
        1
        for g in groups
        for r in (g["parent"], *(g.get("children") or []))
        if r.get("status") == "in_sync"
    )
    n_reopened = sum(1 for r in closed_rows if r["status"] == "reopened")

    return {
        "ok": True,
        "mode": live_cfg.get("mode") or "off",
        "groups": groups,
        "closed_trades": closed_rows,
        "quarantined": list(adopt_result.get("quarantined") or []),
        "counts": {
            "in_sync": n_sync,
            "new": n_new,
            "remove": n_remove,
            "reopened": n_reopened,
            "ig_positions": len(ig_pos_ids),
            "ig_orders": len(ig_wo_ids),
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
        "warnings": list(adopt_result.get("warnings") or []),
    }


def adopt_ig_book(
    session: Session,
    settings: Settings,
    slug: str,
    *,
    mode: str = "replace_open",
) -> dict[str, Any]:
    """
    Fetch IG positions + working orders into the local live ledger.

    Always rebuilds via ``sync_open_book_from_ig`` (same path as every live
    cycle). ``mode`` is accepted for API compatibility and ignored.
    Updates connector order books so the mirror neither cancels nor re-pushes
    imported rows.
    """
    _ = mode
    snap = _fetch_ig_snapshot(session, settings, slug)
    if not snap.get("ok"):
        return snap

    cfg = snap["cfg"]
    connectors = snap["connectors"]
    primary_id = snap["primary_id"]
    positions = snap["positions"]
    raw_orders = snap["raw_orders"]

    with _LOCK:
        sched = _SCHEDULERS.get(slug)
        if sched is not None:
            ledger = sched.ig.ledger
        else:
            state = _read_json(live_state_path(settings, slug), default=None)
            ledger = HedgeLedger.from_state_dict(
                cfg, state if isinstance(state, dict) else None
            )

        # Same rebuild path as every live cycle (ignore additive mode).
        result = sync_open_book_from_ig(
            ledger,
            positions=positions,
            working_orders=raw_orders,
            epic=cfg.epic,
        )

        books_dir = live_dir(settings, slug) / "order_books"
        books_dir.mkdir(parents=True, exist_ok=True)
        connector_ids = [cid for cid, _ in connectors] or [primary_id]
        for cid in connector_ids:
            book_path = books_dir / f"orders_{cid}.json"
            book: dict[str, str] = {}
            for oid, deal_id in (result.get("order_book") or {}).items():
                book[str(oid)] = str(deal_id)
            for oid, order in ledger.working_orders.items():
                did = (order.deal_id or "").strip()
                if did:
                    book[str(oid)] = did
            book_path.write_text(json.dumps(book, indent=2), encoding="utf-8")

        _write_json(live_state_path(settings, slug), ledger.to_state_dict())

    n_pos = len(result.get("imported_positions") or [])
    n_ord = len(result.get("imported_orders") or [])
    n_q = len(result.get("quarantined") or [])
    n_closed = len(result.get("closed") or [])
    n_opened = len(result.get("opened") or [])
    n_imported = len(result.get("imported") or [])
    n_dropped = len(result.get("dropped_orders") or [])
    changed = bool(result.get("changed"))
    msg = f"Open book rebuilt from IG: {n_pos} position(s), {n_ord} order(s)"
    if n_closed:
        msg += f", {n_closed} close(s) recorded"
    if n_q:
        msg += f", {n_q} phantom close(s) quarantined"
    if changed:
        msg += f" · delta +legs {n_opened} +orders {n_imported}"
        if n_dropped:
            msg += f" dropped {n_dropped}"
    msg += "."

    if changed:
        append_sync_log(
            settings,
            slug,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "manual_sync",
                "cycle_id": None,
                "dropped": list(result.get("dropped_orders") or []),
                "imported_orders": list(result.get("imported") or []),
                "opened": list(result.get("opened") or []),
                "closed": list(result.get("closed") or []),
                "quarantined": list(result.get("quarantined") or []),
                "repair": result.get("repair"),
                "warnings": list(result.get("warnings") or []),
                "desync": True,
                "changed": True,
            },
        )
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
    force_llm: bool = True,
) -> dict[str, Any]:
    """
    Force one live/paper cycle for a bot (ignores the candle clock).

    ``force_llm`` (default True): also bypass Adaptive/Fixed LLM schedule so
    Gemini runs when OHLC is usable. Still skips on stale feed / unresolved desync.

    Returns {ok, message, error?, payload?}.
    """
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.interfaces.api.deps import _gemini_api_key

    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    tenant = tenant_svc.get_by_slug(slug)
    if tenant is None:
        return {"ok": False, "message": f"Bot {slug!r} not found.", "error": "not_found"}

    if not tenant.is_trader:
        return {
            "ok": False,
            "message": "This bot is not a trader bot.",
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

    from chatbot.domain.trader_access import trader_settings_as_integration_dict

    integ_cfg = trader_settings_as_integration_dict(tenant)
    gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
    cfg = _build_trader_config(
        live_cfg=live_cfg,
        integ_cfg=integ_cfg,
        primary_ig=connectors[0][1],
        tenant_slug=slug,
        gemini_model=gemini_model,
        system_prompt=str(tenant.prompt or ""),
        market_profile=str(getattr(tenant.config.trader, "market_profile", None) or "cac40"),
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
            payload = sched.run_once(force_llm=bool(force_llm))
            _persist_llm_schedule(sched, settings, slug)
            _write_json(live_state_path(settings, slug), sched.ig.ledger.to_state_dict())
            _append_decision(settings, slug, payload)
            append_sync_log_from_payload(
                settings, slug, payload, source="cycle_manual"
            )
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
                    "force_llm": bool(force_llm),
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
                    "poll_seconds": settings.trader_live_poll_seconds,
                    "tenants_ok": 1,
                    "tenants_failed": 0,
                    "tenants_skipped": 0,
                    "logs": [f"{slug}: manual cycle ok mode={mode} force_llm={bool(force_llm)}"],
                    "trigger": "manual",
                },
            )
            if payload.get("skip_reason") == "market_closed":
                nxt = (payload.get("session") or {}).get("next_open") or "—"
                llm_bit = f"Market closed — idle until {nxt}"
            elif payload.get("skipped"):
                reasons = payload.get("llm_trigger") or []
                ohlc = payload.get("ohlc_feed") or {}
                if ohlc.get("skip_llm"):
                    llm_bit = "LLM skipped (stale/gap OHLC)"
                elif (payload.get("reconcile") or {}).get("desync"):
                    llm_bit = "LLM skipped (book desync)"
                else:
                    llm_bit = (
                        "LLM skipped (trigger quiet)"
                        if not force_llm
                        else "LLM skipped (forced but blocked)"
                    )
                if reasons:
                    llm_bit += f" · {','.join(str(r) for r in reasons)}"
            else:
                llm_bit = (
                    f"LLM ran{' (forced)' if force_llm else ''} · "
                    f"executed={len(payload.get('executed') or [])} "
                    f"rejected={len(payload.get('rejected') or [])}"
                )
            msg = (
                f"Cycle OK ({mode}) · {llm_bit} · "
                f"legs={pnl.get('legs_count', 0)} · "
                f"orders={pnl.get('working_orders_count', 0)} · "
                f"realized={float(pnl.get('realized_session') or 0):.2f}"
            )
            return {"ok": True, "message": msg, "payload": payload}
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


def find_replayable_cycle(
    settings: Settings,
    slug: str,
    *,
    cycle_dir: str | None = None,
) -> dict[str, Any] | None:
    """Return a journal cycle.json dict that has a non-empty LLM decision."""
    journal = live_dir(settings, slug) / "journal"
    if not journal.is_dir():
        return None

    def _load(path: Path) -> dict[str, Any] | None:
        raw = _read_json(path, default=None)
        if not isinstance(raw, dict):
            return None
        dec = raw.get("decision")
        if not isinstance(dec, dict):
            return None
        actions = dec.get("actions")
        if not isinstance(actions, list) or not actions:
            return None
        out = dict(raw)
        if not out.get("cycle_dir"):
            out["cycle_dir"] = path.parent.name
        return out

    wanted = str(cycle_dir or "").strip()
    if wanted:
        path = journal / wanted / "cycle.json"
        return _load(path) if path.is_file() else None

    dirs = sorted(
        (p for p in journal.iterdir() if p.is_dir() and p.name != "market_closed"),
        key=lambda p: p.name,
        reverse=True,
    )
    # Prefer original Gemini decisions — skip cycles that are themselves replays
    # (those may carry already-nudged levels from older gate bugs).
    for prefer_original in (True, False):
        for d in dirs:
            path = d / "cycle.json"
            if not path.is_file():
                continue
            loaded = _load(path)
            if not loaded:
                continue
            if prefer_original and loaded.get("replay_of"):
                continue
            return loaded
    return None


def replay_live_decision(
    session: Session,
    settings: Settings,
    slug: str,
    *,
    cycle_dir: str | None = None,
    session_factory=None,
) -> dict[str, Any]:
    """
    Re-apply a stored LLM decision (no Gemini call) on the current book.

    Dev tool: useful to re-test RiskGate / IG mirror after code changes.
    """
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.trader.models import LlmDecision

    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    tenant = tenant_svc.get_by_slug(slug)
    if tenant is None:
        return {"ok": False, "message": f"Bot {slug!r} not found.", "error": "not_found"}
    if not tenant.is_trader:
        return {
            "ok": False,
            "message": "This bot is not a trader bot.",
            "error": "no_integration",
        }

    stored = find_replayable_cycle(settings, slug, cycle_dir=cycle_dir)
    if not stored:
        return {
            "ok": False,
            "message": "No stored LLM decision to replay"
            + (f" (cycle {cycle_dir})" if cycle_dir else "")
            + ".",
            "error": "no_decision",
        }
    source_cycle = str(stored.get("cycle_dir") or cycle_dir or "")
    decision = LlmDecision.from_dict(stored.get("decision"))

    live_cfg = load_live_config(settings, slug)
    mode = live_cfg["mode"]
    if mode == "off":
        return {
            "ok": False,
            "message": "Bot is Off — arm Live/Paper before replaying a decision.",
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

    from chatbot.domain.trader_access import trader_settings_as_integration_dict

    integ_cfg = trader_settings_as_integration_dict(tenant)
    gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
    cfg = _build_trader_config(
        live_cfg=live_cfg,
        integ_cfg=integ_cfg,
        primary_ig=connectors[0][1],
        tenant_slug=slug,
        gemini_model=gemini_model,
        system_prompt=str(tenant.prompt or ""),
        market_profile=str(getattr(tenant.config.trader, "market_profile", None) or "cac40"),
    )
    # Replay does not call Gemini; placeholder key is fine.
    api_key = "replay-no-llm"

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
            payload = sched.run_once(
                force_llm=False,
                replay_decision=decision,
                replay_of=source_cycle or None,
            )
            _persist_llm_schedule(sched, settings, slug)
            _write_json(live_state_path(settings, slug), sched.ig.ledger.to_state_dict())
            _append_decision(settings, slug, payload)
            append_sync_log_from_payload(
                settings, slug, payload, source="cycle_replay"
            )
            slot = _remember_cycle_slot(slug)
            pnl = sched.ig.ledger.pnl_payload()
            feed_status = _status_from_cycle_payload(payload, base_warnings=warnings)
            notes = list(payload.get("notes") or [])
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
                    "skipped_llm": False,
                    "cycles": prev_cycles + 1,
                    "trigger": "replay",
                    "replay_of": source_cycle,
                    **feed_status,
                },
            )
            _write_json(
                live_worker_status_path(settings),
                {
                    "ok": True,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "poll_seconds": settings.trader_live_poll_seconds,
                    "tenants_ok": 1,
                    "tenants_failed": 0,
                    "tenants_skipped": 0,
                    "logs": [f"{slug}: replay ok of={source_cycle}"],
                    "trigger": "replay",
                },
            )
            note_bit = f" · notes={','.join(notes)}" if notes else ""
            msg = (
                f"Replay OK of {source_cycle} · "
                f"executed={len(payload.get('executed') or [])} "
                f"rejected={len(payload.get('rejected') or [])}"
                f"{note_bit} · "
                f"legs={pnl.get('legs_count', 0)} · "
                f"orders={pnl.get('working_orders_count', 0)}"
            )
            return {"ok": True, "message": msg, "payload": payload}
        except Exception as exc:
            logger.exception("Replay live decision failed for %s", slug)
            return {
                "ok": False,
                "message": f"Replay failed: {exc}",
                "error": "replay_error",
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

    integ_repo = SqlAlchemyTenantRepository(session)
    tenant_svc = TenantService(integ_repo)
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))

    try:
        from chatbot.adapters.persistence.engine import session_factory as make_factory

        factory = make_factory(session.get_bind())
    except Exception:
        factory = None

    for tenant in integ_repo.list_active_traders():
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

        from chatbot.domain.trader_access import trader_settings_as_integration_dict

        integ_cfg = trader_settings_as_integration_dict(tenant)
        gemini_model = tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"
        cfg = _build_trader_config(
            live_cfg=live_cfg,
            integ_cfg=integ_cfg,
            primary_ig=connectors[0][1],
            tenant_slug=slug,
            gemini_model=gemini_model,
            system_prompt=str(tenant.prompt or ""),
            market_profile=str(getattr(tenant.config.trader, "market_profile", None) or "cac40"),
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
                append_sync_log_from_payload(settings, slug, payload, source="cycle")
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
            "poll_seconds": settings.trader_live_poll_seconds,
            "tenants_ok": ok,
            "tenants_failed": failed,
            "tenants_skipped": skipped,
            "logs": logs[-50:],
        },
    )
    return logs
