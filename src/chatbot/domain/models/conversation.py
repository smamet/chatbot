from __future__ import annotations

from dataclasses import dataclass, field

from chatbot.domain.models.message import ChatMessage


@dataclass(slots=True)
class Conversation:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
