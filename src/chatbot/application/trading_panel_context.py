"""Shared context builder for the Trading panel (bot detail tab + legacy /cac40)."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.application.trader_backtest_service import (
    list_runs,
    ohlc_info,
    read_ohlc_sync_status,
    read_ohlc_worker_status,
)
from chatbot.application.trader_live_service import (
    LIVE_CYCLE_SECONDS,
    list_live_cycles,
    load_live_config,
    read_live_book,
    read_live_status,
    read_live_worker_status,
    read_sync_log,
)
from chatbot.application.trader_stream_service import (
    read_stream_quote,
    read_stream_status,
    read_stream_worker_status,
    stream_is_healthy,
)
from chatbot.application.connector_service import ConnectorService
from chatbot.trader.config import TraderConfig
from chatbot.trader.ig_allowance import pick_ig_price_allowance
from chatbot.trader.market_calendar import session_snapshot
from chatbot.trader.profiles import get_profile
from chatbot.config.settings import Settings
from chatbot.domain.models.tenant import Tenant
from chatbot.domain.trader_access import trader_settings_as_integration_dict


def build_trading_panel_context(
    *,
    tenant: Tenant,
    settings: Settings,
    session: Session,
    query_params: Any = None,
) -> dict[str, Any]:
    slug = tenant.slug
    qp = query_params
    integ_cfg = trader_settings_as_integration_dict(tenant)
    profile = get_profile(integ_cfg.get("market_profile"))
    defaults = TraderConfig().to_dict()
    defaults["symbol"] = str(integ_cfg.get("symbol") or profile.default_symbol)
    defaults["epic"] = str(integ_cfg.get("epic") or profile.default_epic)
    if integ_cfg.get("max_open_positions") not in (None, ""):
        try:
            defaults["max_open_positions"] = int(integ_cfg["max_open_positions"])
        except (TypeError, ValueError):
            pass

    live_cfg = load_live_config(settings, slug)
    live_strategy = {**defaults, **(live_cfg.get("strategy") or {})}
    live_status = read_live_status(settings, slug)
    live_worker = read_live_worker_status(settings)
    ohlc_sync = read_ohlc_sync_status(settings, slug)
    dataset = ohlc_info(settings, slug)
    market_session = session_snapshot(
        flatten_lead_minutes=int(live_strategy.get("flatten_lead_minutes") or 30),
        flatten_enabled=bool(live_strategy.get("flatten_before_close", True)),
        calendar_id=profile.calendar_id,
    )
    ig_price_allowance = pick_ig_price_allowance(
        live_status.get("ig_price_allowance")
        if isinstance(live_status.get("ig_price_allowance"), dict)
        else None,
        ohlc_sync.get("ig_price_allowance")
        if isinstance(ohlc_sync.get("ig_price_allowance"), dict)
        else None,
    )
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    ig_config = conn_svc.get_ig_config(tenant.id)
    ig_list = conn_svc.list_ig(tenant.id)
    ig_connectors = [
        {
            "id": c.id,
            "name": str(c.config.get("name") or f"IG #{c.id}"),
            "acc_type": str(c.config.get("acc_type") or "DEMO").upper(),
            "epic": str(c.config.get("epic") or "—"),
            "active": c.active,
            "selected": c.id in (live_cfg.get("ig_connector_ids") or []),
        }
        for c in ig_list
    ]
    last_cycle = live_status.get("last_cycle_at") or live_status.get("finished_at")
    worker_finished = live_worker.get("finished_at")
    stale = False
    awaiting_first_cycle = False
    if live_cfg["mode"] != "off":
        heartbeat = worker_finished or last_cycle
        if heartbeat:
            try:
                from datetime import datetime, timezone

                finished = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))
                age = (
                    datetime.now(timezone.utc) - finished.astimezone(timezone.utc)
                ).total_seconds()
                poll = max(60, int(settings.trader_live_poll_seconds or 60))
                stale = age > poll * 3
            except Exception:
                stale = True
        else:
            stale = True
        awaiting_first_cycle = not bool(last_cycle) and not stale

    stream_status = read_stream_status(settings, slug)
    stream_worker = read_stream_worker_status(settings)
    stream_quote = read_stream_quote(settings, slug)
    stream_worker_down = False
    if live_cfg["mode"] != "off":
        try:
            from datetime import datetime, timezone

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
    stream_healthy = (
        stream_is_healthy(
            stream_status,
            dealing_open=bool(market_session.get("dealing_open")) if market_session else True,
        )
        and not stream_worker_down
    )
    stream_badge = "off"
    if live_cfg["mode"] != "off":
        if stream_worker_down:
            stream_badge = "down"
        elif stream_status.get("stale") or not stream_status.get("connected"):
            stream_badge = "stale"
        elif stream_healthy:
            stream_badge = "ok"
        else:
            stream_badge = "stale"

    def _q(key: str) -> str | None:
        if qp is None:
            return None
        try:
            return qp.get(key)
        except Exception:
            return None

    return {
        "runs": list_runs(settings, slug),
        "ohlc": dataset,
        "ohlc_path": dataset["path"],
        "ohlc_exists": dataset["exists"],
        "bot_symbol": defaults["symbol"],
        "bot_epic": defaults["epic"],
        "default_config": defaults,
        "upload_error": _q("upload_error"),
        "upload_ok": _q("upload_ok"),
        "sync_error": _q("sync_error"),
        "sync_ok": _q("sync_ok"),
        "live_ok": _q("live_ok"),
        "live_error": _q("live_error"),
        "ig_connector_ready": bool(ig_config) or any(c["active"] for c in ig_connectors),
        "ohlc_sync_status": ohlc_sync,
        "ohlc_worker_status": read_ohlc_worker_status(settings),
        "trader_ohlc_poll_seconds": settings.trader_ohlc_poll_seconds,
        "ig_price_allowance": ig_price_allowance,
        "live_config": live_cfg,
        "live_strategy": live_strategy,
        "live_mode": live_cfg["mode"],
        "live_status": live_status,
        "live_worker_status": live_worker,
        "live_stale": stale,
        "live_awaiting_first_cycle": awaiting_first_cycle,
        "stream_status": stream_status,
        "stream_worker_status": stream_worker,
        "stream_quote": stream_quote,
        "stream_healthy": stream_healthy,
        "stream_worker_down": stream_worker_down,
        "stream_badge": stream_badge,
        "market_session": market_session,
        "ig_connectors": ig_connectors,
        "trader_live_poll_seconds": settings.trader_live_poll_seconds,
        "live_cycle_seconds": LIVE_CYCLE_SECONDS,
        "live_cycles": list_live_cycles(settings, slug, limit=3),
        "live_book": read_live_book(settings, slug),
        "sync_log_count": len(read_sync_log(settings, slug, limit=200)),
        "market_profile": profile.id,
        "market_profile_label": profile.label,
        "dev_mode": bool(settings.dev_mode),
    }
