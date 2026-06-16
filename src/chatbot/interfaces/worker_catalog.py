from __future__ import annotations

import argparse
import logging
import time

from chatbot.adapters.persistence.disk_usage_repository import SqlAlchemyDiskUsageRepository
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.disk_snapshot_service import DiskSnapshotService
from chatbot.application.disk_usage_service import DiskUsageService
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.application.erpnext_catalog_sync_service import run_due_catalog_syncs
from chatbot.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_once(settings, factory) -> list[str]:
    logs: list[str] = []
    with factory() as session:
        logs.extend(run_due_catalog_syncs(session, settings=settings))
        disk_svc = DiskSnapshotService(
            settings=settings,
            disk_repo=SqlAlchemyDiskUsageRepository(session),
            tenant_repo=SqlAlchemyTenantRepository(session),
            disk_usage=DiskUsageService(settings),
        )
        logs.extend(disk_svc.record_all_if_due())
        if logs:
            session.commit()
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ERPNext catalog snapshots into RAG.")
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

    interval = max(1, settings.catalog_poll_seconds)
    logger.info("Catalog worker started (poll every %ss)", interval)
    while True:
        try:
            logs = run_once(settings, factory)
            for line in logs:
                logger.info("%s", line)
        except Exception:
            logger.exception("Catalog worker poll cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
