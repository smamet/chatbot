from __future__ import annotations

import argparse
import logging
import time

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.application.trader_backtest_service import run_due_ig_ohlc_syncs
from chatbot.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_once(settings, factory) -> list[str]:
    with factory() as session:
        return run_due_ig_ohlc_syncs(session, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CAC40 OHLC from IG connectors.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll cycle and exit.",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)

    if args.once:
        logs = run_once(settings, factory)
        for line in logs:
            logger.info("%s", line)
        return

    interval = max(1, settings.trader_ohlc_poll_seconds)
    logger.info("CAC40 OHLC worker started (poll every %ss)", interval)
    while True:
        try:
            logs = run_once(settings, factory)
            for line in logs:
                logger.info("%s", line)
        except Exception:
            logger.exception("CAC40 OHLC worker poll cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
