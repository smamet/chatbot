from __future__ import annotations

import logging
from collections.abc import Callable

from evenor.domain.models.hook import HookEvent

logger = logging.getLogger(__name__)

HookHandler = Callable[[HookEvent], None]


class HookRegistry:
    def __init__(self) -> None:
        self._handlers: list[tuple[str, HookHandler]] = []

    def register(self, type_prefix: str, handler: HookHandler) -> None:
        self._handlers.append((type_prefix, handler))

    def dispatch(self, hook: HookEvent) -> None:
        for prefix, handler in self._handlers:
            if hook.type == prefix or hook.type.startswith(f"{prefix}."):
                handler(hook)
                return
        logger.warning("No handler for hook type %s (id=%s)", hook.type, hook.id)
