from __future__ import annotations

import logging
import time

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.automation.handlers import dispatch_hook
from chatbot.config.settings import get_settings
from chatbot.domain.models.hook import HookStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_once() -> int:
    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    processed = 0
    with factory() as session:
        repo = SqlAlchemyHookEventRepository(session, tenant_id=None)
        hooks = repo.claim_pending(limit=20)
        for hook in hooks:
            try:
                dispatch_hook(session, hook)
                repo.update_status(hook.id, status=HookStatus.DONE)
                processed += 1
            except Exception as e:
                logger.exception("Hook %s failed", hook.id)
                repo.update_status(
                    hook.id,
                    status=HookStatus.FAILED,
                    error=str(e),
                    increment_attempts=True,
                )
        session.commit()
    return processed


def main() -> None:
    settings = get_settings()
    interval = max(1, settings.hook_poll_seconds)
    logger.info("Automation worker started (poll every %ss)", interval)
    while True:
        try:
            n = run_once()
            if n:
                logger.info("Processed %s hook(s)", n)
        except Exception:
            logger.exception("Worker poll cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
