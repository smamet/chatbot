"""Lightstreamer orchestration: OHLC quotes/bars, TRADE wake-up reconcile, health."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from chatbot.application.trader_backtest_service import default_ohlc_path, trader_root
from chatbot.application.trader_live_service import (
    live_cycle_lock,
    live_dir,
    live_state_path,
    load_live_config,
    normalize_live_mode,
    sync_open_book_from_ig,
)
from chatbot.config.settings import Settings
from chatbot.trader.config import TraderConfig
from chatbot.trader.hedge_ledger import HedgeLedger
from chatbot.trader.ig_connector import IgConnector
from chatbot.trader.ig_session_cache import login_with_shared_cache
from chatbot.trader.ig_stream_probe import TickBar
from chatbot.trader.ig_stream_service import IgStreamService
from chatbot.trader.ohlc_store import append_bars, connects_15m, is_natural_session_break, load_ohlc_csv

logger = logging.getLogger(__name__)

STREAM_TICK_STALE_SECONDS = 120.0
STREAM_DISCONNECT_STALE_SECONDS = 30.0
STREAM_TRADE_DEBOUNCE_SECONDS = 0.35
STREAM_SUPERVISOR_LOOP_SECONDS = 5.0
STREAM_REST_RECONCILE_MINUTES = 15.0
STREAM_BOOK_FRESH_SECONDS = 5.0
# Rate-limit REST gap repairs so reconnect flaps do not burn IG /prices allowance.
STREAM_GAP_REPAIR_RETRY_SECONDS = 60.0


def stream_status_path(settings: Settings, slug: str) -> Path:
    return trader_root(settings, slug) / "stream_status.json"


def stream_quote_path(settings: Settings, slug: str) -> Path:
    return trader_root(settings, slug) / "stream_quote.json"


def stream_worker_status_path(settings: Settings) -> Path:
    return Path(settings.data_root) / "trader" / "stream_worker_status.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_stream_status(settings: Settings, slug: str, payload: dict[str, Any]) -> None:
    _write_json(stream_status_path(settings, slug), payload)


def read_stream_status(settings: Settings, slug: str) -> dict[str, Any]:
    return _read_json(stream_status_path(settings, slug))


def write_stream_quote(settings: Settings, slug: str, payload: dict[str, Any]) -> None:
    _write_json(stream_quote_path(settings, slug), payload)


def read_stream_quote(settings: Settings, slug: str) -> dict[str, Any]:
    return _read_json(stream_quote_path(settings, slug))


def write_stream_worker_status(settings: Settings, payload: dict[str, Any]) -> None:
    _write_json(stream_worker_status_path(settings), payload)


def read_stream_worker_status(settings: Settings) -> dict[str, Any]:
    return _read_json(stream_worker_status_path(settings))


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def evaluate_stream_stale(
    status: dict[str, Any],
    *,
    dealing_open: bool,
    now: datetime | None = None,
    tick_stale_seconds: float = STREAM_TICK_STALE_SECONDS,
    disconnect_stale_seconds: float = STREAM_DISCONNECT_STALE_SECONDS,
) -> tuple[bool, str | None]:
    """Return (stale, reason) for a per-bot stream_status payload."""
    now = now or datetime.now(timezone.utc)
    connected = bool(status.get("connected"))
    if not connected:
        last_hb = _parse_iso(str(status.get("last_heartbeat_at") or ""))
        # Prefer explicit disconnect age from last_reconnect / heartbeat.
        ref = _parse_iso(str(status.get("disconnected_at") or "")) or last_hb
        if ref is None:
            return True, "disconnected"
        age = (now - ref).total_seconds()
        if age > disconnect_stale_seconds:
            return True, "disconnected"
        return True, "disconnected"

    market_state = str(status.get("market_state") or "").strip().upper()
    tradeable = dealing_open and (not market_state or market_state in ("TRADEABLE", "AVAILABLE"))
    if not tradeable:
        return False, None

    last_tick = _parse_iso(str(status.get("last_tick_at") or ""))
    if last_tick is None:
        # Connected but never received a tick — give a short grace via heartbeat.
        last_hb = _parse_iso(str(status.get("last_heartbeat_at") or ""))
        if last_hb is None:
            return True, "no_ticks"
        if (now - last_hb).total_seconds() > tick_stale_seconds:
            return True, "no_ticks"
        return False, None
    if (now - last_tick).total_seconds() > tick_stale_seconds:
        return True, "no_ticks"
    return False, None


def stream_is_healthy(status: dict[str, Any], *, dealing_open: bool = True) -> bool:
    if not status:
        return False
    if status.get("stale"):
        return False
    stale, _reason = evaluate_stream_stale(status, dealing_open=dealing_open)
    return bool(status.get("connected")) and not stale


def stream_book_reconcile_is_fresh(
    status: dict[str, Any],
    *,
    max_age_seconds: float = STREAM_BOOK_FRESH_SECONDS,
    now: datetime | None = None,
) -> bool:
    now = now or datetime.now(timezone.utc)
    if not stream_is_healthy(status, dealing_open=True):
        return False
    ts = _parse_iso(str(status.get("last_reconcile_at") or ""))
    if ts is None:
        return False
    return (now - ts).total_seconds() <= max_age_seconds


def local_ohlc_is_caught_up(
    settings: Settings,
    slug: str,
    *,
    timezone_name: str = "Europe/Paris",
    now: datetime | None = None,
) -> bool:
    """True when local 15m CSV reaches the last closed bar (Data Sync / REST top-up done).

    Cross-day / weekend holes are *not* treated as caught-up — those still need a
    REST fill when the session reopens. Only ``last >= expected`` clears the flag.
    """
    from chatbot.trader.live_ohlc_feed import expected_last_closed_15m

    path = default_ohlc_path(settings, slug)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        existing = load_ohlc_csv(path, timezone=timezone_name)
        if existing.empty:
            return False
        last = pd.Timestamp(existing.index[-1])
        expected = expected_last_closed_15m(now=now, tz=timezone_name)
        return bool(last >= expected)
    except Exception as exc:
        logger.warning("local OHLC catch-up check failed slug=%s: %s", slug, exc)
        return False


def append_closed_stream_bar(
    settings: Settings,
    slug: str,
    bar: TickBar,
    *,
    timezone_name: str = "Europe/Paris",
) -> dict[str, Any]:
    """Append one closed synthetic 15m bar (bucket_start = bar open ts)."""
    path = default_ohlc_path(settings, slug)
    bucket = bar.bucket_start
    if bucket.tzinfo is None:
        ts = pd.Timestamp(bucket, tz="UTC").tz_convert(timezone_name)
    else:
        ts = pd.Timestamp(bucket).tz_convert(timezone_name)
    # Closed bar is labeled at bucket start (same convention as IG 15m).
    row = pd.DataFrame(
        {
            "open": [float(bar.open)],
            "high": [float(bar.high)],
            "low": [float(bar.low)],
            "close": [float(bar.close)],
            "volume": [float(bar.ticks)],
        },
        index=pd.DatetimeIndex([ts], name="ts"),
    )

    out: dict[str, Any] = {"added": 0, "path": str(path), "ts": str(ts)}
    if not path.exists():
        out["error"] = "ohlc_csv_missing"
        return out
    try:
        existing = load_ohlc_csv(path, timezone=timezone_name)
        if not existing.empty:
            last = pd.Timestamp(existing.index[-1])
            if ts <= last:
                out["skipped"] = "already_have"
                return out
            if not (
                connects_15m(last, ts) or is_natural_session_break(last, ts)
            ):
                out["error"] = "gap"
                out["last"] = str(last)
                return out
        append_bars(path, row, require_contiguous=True)
        out["added"] = 1
    except Exception as exc:
        out["error"] = str(exc)
        logger.warning("stream bar append failed slug=%s: %s", slug, exc)
    return out


def reconcile_open_book_under_lock(
    settings: Settings,
    slug: str,
    *,
    ig_config: dict[str, Any],
    cfg: TraderConfig,
    blocking: bool = True,
) -> dict[str, Any]:
    """REST list positions+WOs → replace_open under the live cycle lock."""
    result: dict[str, Any] = {"ok": False, "acquired": False, "changed": False}
    with live_cycle_lock(settings, slug, blocking=blocking) as acquired:
        result["acquired"] = bool(acquired)
        if not acquired:
            result["error"] = "lock_busy"
            return result
        ig = IgConnector(cfg, dry_run=False)
        try:
            login_with_shared_cache(ig)
            positions = ig.list_open_positions()
            working_orders = ig.list_working_orders()
            state = _read_json(live_state_path(settings, slug))
            ledger = HedgeLedger.from_state_dict(
                cfg, state if isinstance(state, dict) else None
            )
            sync = sync_open_book_from_ig(
                ledger,
                positions=positions,
                working_orders=working_orders,
                epic=cfg.epic,
            )
            books_dir = live_dir(settings, slug) / "order_books"
            books_dir.mkdir(parents=True, exist_ok=True)
            primary_id = int(ig_config.get("_connector_id") or 0)
            book_name = f"orders_{primary_id}.json" if primary_id else "orders_primary.json"
            book: dict[str, str] = {}
            for oid, deal_id in (sync.get("order_book") or {}).items():
                book[str(oid)] = str(deal_id)
            for oid, order in ledger.working_orders.items():
                did = (order.deal_id or "").strip()
                if did:
                    book[str(oid)] = did
            (books_dir / book_name).write_text(
                json.dumps(book, indent=2), encoding="utf-8"
            )
            _write_json(live_state_path(settings, slug), ledger.to_state_dict())
            result.update(
                {
                    "ok": True,
                    "changed": bool(sync.get("changed")),
                    "imported": list(sync.get("imported") or []),
                    "closed": list(sync.get("closed") or []),
                    "opened": list(sync.get("opened") or []),
                    "warnings": list(sync.get("warnings") or []),
                }
            )
        except Exception as exc:
            logger.exception("stream book reconcile failed slug=%s", slug)
            result["error"] = str(exc)
        finally:
            ig.close()
    return result


def repair_ohlc_gap_via_rest(
    settings: Settings,
    slug: str,
    *,
    ig_config: dict[str, Any],
    cfg: TraderConfig,
) -> dict[str, Any]:
    """One REST OHLC top-up after stream gap/recovery (burns allowance)."""
    from chatbot.trader.live_ohlc_feed import top_up_csv_from_connector

    path = default_ohlc_path(settings, slug)
    ig = IgConnector(cfg, dry_run=False)
    try:
        login_with_shared_cache(ig)
        return top_up_csv_from_connector(
            path,
            ig,
            timezone_name=str(cfg.data_timezone or "Europe/Paris"),
        )
    except Exception as exc:
        logger.warning("stream OHLC gap repair failed slug=%s: %s", slug, exc)
        return {"added": 0, "error": str(exc)}
    finally:
        ig.close()


class BotStreamRuntime:
    """Per-bot Lightstreamer runtime: quotes, closed bars, TRADE wake-up reconcile."""

    def __init__(
        self,
        *,
        settings: Settings,
        slug: str,
        mode: str,
        ig_config: dict[str, Any],
        cfg: TraderConfig,
        enable_trade_reconcile: bool = True,
    ) -> None:
        self.settings = settings
        self.slug = slug
        self.mode = normalize_live_mode(mode)
        self.ig_config = dict(ig_config)
        self.cfg = cfg
        self.enable_trade_reconcile = enable_trade_reconcile
        self._service: IgStreamService | None = None
        self._debounce_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._need_gap_repair = False
        self._was_tick_stale = False
        self._last_gap_repair_mono = 0.0
        self._gap_repair_last_error: str | None = None
        self._status: dict[str, Any] = {
            "ok": False,
            "connected": False,
            "slug": slug,
            "mode": self.mode,
            "epic": cfg.epic,
            "account_id": str(ig_config.get("account_id") or ""),
        }

    def start(self) -> None:
        ig = IgConnector(self.cfg, dry_run=False)
        try:
            session = login_with_shared_cache(ig)
        finally:
            # Keep tokens; close HTTP client only.
            ig.close()
        endpoint = session.lightstreamer_endpoint
        account_id = session.account_id or str(self.ig_config.get("account_id") or "")
        if not endpoint or not account_id:
            self._status.update(
                {
                    "ok": False,
                    "error": "missing_ls_endpoint_or_account",
                    "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_stream_status(self.settings, self.slug, self._status)
            return

        def _relogin() -> tuple[str, str, str] | None:
            tmp = IgConnector(self.cfg, dry_run=False)
            try:
                from chatbot.trader.ig_session_cache import (
                    invalidate_cached_session,
                    session_cache_key,
                )

                invalidate_cached_session(
                    session_cache_key(
                        api_key=self.cfg.ig_api_key or "",
                        username=self.cfg.ig_username or "",
                        account_id=self.cfg.ig_account_id or "",
                        acc_type=self.cfg.ig_acc_type or "DEMO",
                    )
                )
                s = login_with_shared_cache(tmp, force=True)
                return s.cst, s.xst, s.lightstreamer_endpoint
            finally:
                tmp.close()

        self._service = IgStreamService(
            endpoint=endpoint,
            account_id=account_id,
            cst=session.cst,
            xst=session.xst,
            epic=self.cfg.epic,
            on_tick=self._on_tick,
            on_bar_closed=self._on_bar_closed,
            on_trade_wakeup=self._on_trade_wakeup if self.enable_trade_reconcile else None,
            on_status=self._on_ls_status,
            on_need_relogin=_relogin,
        )
        self._status.update(
            {
                "account_id": account_id,
                "epic": self.cfg.epic,
                "lightstreamer_endpoint": endpoint,
            }
        )
        self._service.start()
        # Baseline book on connect (live dealing only).
        if self.enable_trade_reconcile:
            self._schedule_reconcile(delay=0.5)
        # Auto-repair: if CSV is behind after (re)start, schedule a REST top-up
        # on the next healthy heartbeat instead of waiting for a bar-gap event.
        if not local_ohlc_is_caught_up(
            self.settings,
            self.slug,
            timezone_name=str(self.cfg.data_timezone or "Europe/Paris"),
        ):
            self._need_gap_repair = True
        self.heartbeat()

    def stop(self) -> None:
        if self._debounce_timer:
            self._debounce_timer.cancel()
            self._debounce_timer = None
        if self._service:
            self._service.stop()
            self._service = None
        self._status["connected"] = False
        self._status["disconnected_at"] = datetime.now(timezone.utc).isoformat()
        self.heartbeat()

    def heartbeat(self, *, dealing_open: bool = True) -> dict[str, Any]:
        svc = self._service
        now = datetime.now(timezone.utc).isoformat()
        if svc is not None:
            self._status.update(
                {
                    "connected": svc.connected,
                    "ls_status": svc.status,
                    "last_tick_at": svc.last_tick_at,
                    "last_bar_closed_at": svc.last_bar_closed_at,
                    "last_trade_at": svc.last_trade_at,
                    "ticks_total": svc.ticks_total,
                    "bars_closed_total": svc.bars_closed_total,
                    "trade_events_total": svc.trade_events_total,
                    "reconnect_count": svc.reconnect_count,
                    "market_state": svc.last_quote.market_state,
                }
            )
            svc_err = (svc.last_error or "").strip()
            if svc_err:
                self._status["error"] = svc_err
            elif not self._need_gap_repair:
                self._status.pop("error", None)
            if not svc.connected and not self._status.get("disconnected_at"):
                self._status["disconnected_at"] = now
            if svc.connected:
                self._status.pop("disconnected_at", None)
        self._status["last_heartbeat_at"] = now
        tick_stale, reason = evaluate_stream_stale(
            self._status, dealing_open=dealing_open
        )

        # After a tick/disconnect outage, CSV may have holes — queue auto-repair.
        if self._was_tick_stale and not tick_stale and svc and svc.connected:
            self._need_gap_repair = True

        # Auto-repair while ticks are healthy (does not require stale→ok edge).
        # Previous logic forced stale=True when _need_gap_repair, which blocked
        # the recovery path forever after one failed REST top-up.
        if (
            self._need_gap_repair
            and not tick_stale
            and svc
            and svc.connected
            and dealing_open
        ):
            self._maybe_auto_repair_gap()

        stale = bool(tick_stale)
        if self._need_gap_repair and dealing_open:
            stale = True
            if self._gap_repair_last_error:
                reason = "gap_repair_failed"
                self._status["error"] = f"gap_repair:{self._gap_repair_last_error}"
            else:
                reason = reason or "gap_repair_pending"

        self._was_tick_stale = bool(tick_stale)
        self._status["stale"] = stale
        self._status["stale_reason"] = reason
        self._status["need_gap_repair"] = bool(self._need_gap_repair)
        self._status["ok"] = bool(self._status.get("connected")) and not stale
        write_stream_status(self.settings, self.slug, self._status)
        return dict(self._status)

    def _clear_gap_repair(self, *, result: str) -> None:
        self._need_gap_repair = False
        self._gap_repair_last_error = None
        self._status.pop("error", None)
        self._status["gap_repair_at"] = datetime.now(timezone.utc).isoformat()
        self._status["gap_repair_result"] = result

    def _maybe_auto_repair_gap(self) -> bool:
        """
        Clear OHLC gaps when the stream is healthy again.

        1. If Data-tab Sync (or another top-up) already caught the CSV up → clear.
        2. Else rate-limited REST ``/prices`` top-up.
        """
        tz = str(self.cfg.data_timezone or "Europe/Paris")
        if local_ohlc_is_caught_up(self.settings, self.slug, timezone_name=tz):
            self._clear_gap_repair(result="already_caught_up")
            logger.info("stream gap repair skipped (CSV caught up) slug=%s", self.slug)
            return True

        now_mono = time.monotonic()
        if (
            self._last_gap_repair_mono
            and (now_mono - self._last_gap_repair_mono) < STREAM_GAP_REPAIR_RETRY_SECONDS
        ):
            return False

        self._last_gap_repair_mono = now_mono
        repair = repair_ohlc_gap_via_rest(
            self.settings, self.slug, ig_config=self.ig_config, cfg=self.cfg
        )
        if repair.get("error"):
            err = str(repair["error"])
            self._gap_repair_last_error = err
            self._status["error"] = f"gap_repair:{err}"
            self._status["gap_repair_at"] = datetime.now(timezone.utc).isoformat()
            self._status["gap_repair_result"] = "failed"
            logger.warning(
                "stream auto gap repair failed slug=%s: %s (retry in %.0fs)",
                self.slug,
                err,
                STREAM_GAP_REPAIR_RETRY_SECONDS,
            )
            return False

        added = int(repair.get("added") or 0)
        # Trust REST success, or accept CSV already contiguous after Sync race.
        if added or local_ohlc_is_caught_up(
            self.settings, self.slug, timezone_name=tz
        ):
            self._clear_gap_repair(result=f"added={added}")
            logger.info(
                "stream auto gap repair ok slug=%s added=%s", self.slug, added
            )
            return True

        # REST returned no bars but CSV still behind — keep trying.
        self._gap_repair_last_error = "no_bars_added"
        self._status["error"] = "gap_repair:no_bars_added"
        self._status["gap_repair_at"] = datetime.now(timezone.utc).isoformat()
        self._status["gap_repair_result"] = "no_bars_added"
        return False

    def _on_ls_status(self, status: str) -> None:
        self._status["ls_status"] = status
        self._status["connected"] = "CONNECTED" in (status or "")
        if self._status["connected"]:
            self._status.pop("disconnected_at", None)
        else:
            self._status["disconnected_at"] = datetime.now(timezone.utc).isoformat()

    def _on_tick(self, quote: dict[str, Any]) -> None:
        write_stream_quote(self.settings, self.slug, quote)

    def _on_bar_closed(self, bar: TickBar, quote: dict[str, Any]) -> None:
        tz = str(self.cfg.data_timezone or "Europe/Paris")
        result = append_closed_stream_bar(
            self.settings,
            self.slug,
            bar,
            timezone_name=tz,
        )
        if result.get("error") == "gap":
            self._need_gap_repair = True
            # Bypass backoff for the immediate bar-close path (one shot).
            self._last_gap_repair_mono = 0.0
            if self._maybe_auto_repair_gap():
                result = append_closed_stream_bar(
                    self.settings,
                    self.slug,
                    bar,
                    timezone_name=tz,
                )
                if result.get("error") == "gap":
                    # Still non-contiguous after top-up — keep pending for heartbeat.
                    self._need_gap_repair = True
                elif result.get("added") or result.get("skipped") == "already_have":
                    self._clear_gap_repair(
                        result="bar_appended"
                        if result.get("added")
                        else "already_have"
                    )
        write_stream_quote(self.settings, self.slug, quote)

    def _on_trade_wakeup(self, _fields: dict[str, Any]) -> None:
        self._schedule_reconcile(delay=STREAM_TRADE_DEBOUNCE_SECONDS)

    def _schedule_reconcile(self, *, delay: float) -> None:
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(delay, self._run_reconcile)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _run_reconcile(self) -> None:
        result = reconcile_open_book_under_lock(
            self.settings,
            self.slug,
            ig_config=self.ig_config,
            cfg=self.cfg,
            blocking=True,
        )
        self._status["last_reconcile_at"] = datetime.now(timezone.utc).isoformat()
        self._status["last_reconcile_ok"] = bool(result.get("ok"))
        if result.get("error"):
            self._status["error"] = f"reconcile:{result['error']}"
        write_stream_status(self.settings, self.slug, self._status)


def discover_armed_stream_bots(
    session: Session, settings: Settings
) -> list[dict[str, Any]]:
    """Return armed (mode=live) trader bots with primary IG config."""
    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.application.connector_service import ConnectorService

    tenants = SqlAlchemyTenantRepository(session).list_active_traders()
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    out: list[dict[str, Any]] = []
    for tenant in tenants:
        live_cfg = load_live_config(settings, tenant.slug)
        mode = normalize_live_mode(live_cfg.get("mode"))
        if mode != "live":
            continue
        selected = [int(i) for i in (live_cfg.get("ig_connector_ids") or [])]
        ig_list = conn_svc.list_ig(tenant.id)
        primary = None
        primary_id = None
        if selected:
            for c in ig_list:
                if c.id in selected and c.active:
                    primary = dict(c.config)
                    primary_id = c.id
                    break
        if primary is None:
            cfg = conn_svc.get_ig_config(tenant.id)
            if cfg:
                primary = dict(cfg)
                for c in ig_list:
                    if c.active:
                        primary_id = c.id
                        break
        if not primary:
            continue
        primary["_connector_id"] = primary_id or 0
        from chatbot.domain.trader_access import trader_settings_as_integration_dict
        from chatbot.trader.profiles import get_profile

        integ = trader_settings_as_integration_dict(tenant)
        profile = get_profile(integ.get("market_profile"))
        strategy = {**(live_cfg.get("strategy") or {})}
        epic = str(
            strategy.get("epic")
            or primary.get("epic")
            or integ.get("epic")
            or profile.default_epic
        )
        cfg = TraderConfig(
            ig_api_key=str(primary.get("api_key") or ""),
            ig_username=str(primary.get("username") or ""),
            ig_password=str(primary.get("password") or ""),
            ig_account_id=str(primary.get("account_id") or ""),
            ig_acc_type=str(primary.get("acc_type") or "DEMO").upper(),
            epic=epic,
            symbol=str(strategy.get("symbol") or integ.get("symbol") or profile.default_symbol),
            data_timezone=str(strategy.get("data_timezone") or "Europe/Paris"),
        )
        out.append(
            {
                "slug": tenant.slug,
                "tenant_id": tenant.id,
                "mode": mode,
                "ig_config": primary,
                "cfg": cfg,
                "calendar_id": profile.calendar_id,
            }
        )
    return out
