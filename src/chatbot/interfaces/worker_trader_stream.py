"""Long-lived Lightstreamer worker: live OHLC ticks + TRADE wake-up book sync."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.application.trader_stream_service import (
    STREAM_SUPERVISOR_LOOP_SECONDS,
    BotStreamRuntime,
    discover_armed_stream_bots,
    write_stream_worker_status,
)
from chatbot.config.settings import get_settings
from chatbot.trader.market_calendar import session_snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dealing_open(calendar_id: str | None) -> bool:
    try:
        snap = session_snapshot(calendar_id=calendar_id or "euronext_fr40")
        return bool(snap.get("dealing_open"))
    except Exception:
        return True


def run_supervisor(settings, factory, *, once: bool = False) -> None:
    runtimes: dict[str, BotStreamRuntime] = {}
    loop_s = max(2.0, float(STREAM_SUPERVISOR_LOOP_SECONDS))
    refresh_every = max(loop_s, 30.0)
    last_refresh = 0.0

    def refresh_bots() -> None:
        nonlocal last_refresh
        with factory() as session:
            bots = discover_armed_stream_bots(session, settings)
        wanted = {b["slug"] for b in bots}
        for slug in list(runtimes.keys()):
            if slug not in wanted:
                logger.info("Stopping stream for disarmed bot %s", slug)
                runtimes.pop(slug).stop()
        for bot in bots:
            slug = bot["slug"]
            if slug in runtimes:
                continue
            logger.info("Starting stream for %s epic=%s", slug, bot["cfg"].epic)
            rt = BotStreamRuntime(
                settings=settings,
                slug=slug,
                mode=bot["mode"],
                ig_config=bot["ig_config"],
                cfg=bot["cfg"],
                enable_trade_reconcile=True,
            )
            rt.start()
            runtimes[slug] = rt
            # stash calendar for heartbeat
            rt._calendar_id = bot.get("calendar_id")  # type: ignore[attr-defined]
        last_refresh = time.monotonic()

    refresh_bots()
    while True:
        now_mono = time.monotonic()
        if now_mono - last_refresh >= refresh_every:
            try:
                refresh_bots()
            except Exception:
                logger.exception("stream bot refresh failed")

        ok = 0
        failed = 0
        logs: list[str] = []
        for slug, rt in list(runtimes.items()):
            try:
                cal = getattr(rt, "_calendar_id", None)
                status = rt.heartbeat(dealing_open=_dealing_open(cal))
                if status.get("ok"):
                    ok += 1
                else:
                    failed += 1
                    logs.append(
                        f"{slug}: stale={status.get('stale_reason') or status.get('error') or 'not_ok'}"
                    )
            except Exception as exc:
                failed += 1
                logs.append(f"{slug}: heartbeat_error:{exc}")
                logger.exception("stream heartbeat failed for %s", slug)

        write_stream_worker_status(
            settings,
            {
                "ok": failed == 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "poll_seconds": loop_s,
                "tenants_ok": ok,
                "tenants_failed": failed,
                "tenants_skipped": 0,
                "tenants_total": len(runtimes),
                "logs": logs[-50:],
            },
        )
        if once:
            for rt in runtimes.values():
                rt.stop()
            return
        time.sleep(loop_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="IG Lightstreamer OHLC + TRADE sync worker.")
    parser.add_argument("--once", action="store_true", help="One supervisor tick then exit.")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    logger.info("Trader stream worker started")
    try:
        run_supervisor(settings, factory, once=args.once)
    except KeyboardInterrupt:
        logger.info("Trader stream worker stopped")


if __name__ == "__main__":
    main()
