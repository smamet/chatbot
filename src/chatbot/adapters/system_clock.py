from __future__ import annotations

from datetime import UTC, datetime

from chatbot.domain.contracts.clock import Clock


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
