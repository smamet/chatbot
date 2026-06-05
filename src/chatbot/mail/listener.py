from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class ImapMailListener:
    """Poll IMAP inbox and enqueue drafts (stub — wire credentials in phase 2)."""

    def __init__(self, *, poll_seconds: int = 60) -> None:
        self._poll_seconds = poll_seconds

    def run_forever(self) -> None:
        logger.info("Mail worker started (stub). Configure IMAP_* env vars to enable polling.")
        while True:
            time.sleep(self._poll_seconds)
