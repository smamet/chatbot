from __future__ import annotations

from collections.abc import Callable


class ProgressLog:
    def __init__(self, emit: Callable[[str], None] | None = None) -> None:
        self._emit = emit
        self.messages: list[str] = []

    def step(self, message: str) -> None:
        text = message.strip()
        if not text:
            return
        self.messages.append(text)
        if self._emit is not None:
            self._emit(text)
