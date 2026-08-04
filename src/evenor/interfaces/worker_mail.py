from __future__ import annotations

import argparse
import logging
import time

from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.config.settings import get_settings
from evenor.mail.listener import run_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll IMAP inboxes for inbound email.")
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
        n = run_once(factory, settings)
        logger.info("Processed %s mail(s)", n)
        return

    interval = max(1, settings.mail_poll_seconds)
    logger.info("Mail worker started (poll every %ss)", interval)
    while True:
        try:
            n = run_once(factory, settings)
            if n:
                logger.info("Processed %s mail(s)", n)
        except Exception:
            logger.exception("Worker poll cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
