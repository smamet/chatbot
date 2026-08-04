from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from evenor.domain.models.context_debug import ContextDebugInfo


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    context_debug: ContextDebugInfo | None = None
    created_at: datetime | None = None
