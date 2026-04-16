from __future__ import annotations

from typing import Protocol, runtime_checkable

from chatbot.domain.models.message import ChatMessage


@runtime_checkable
class ConversationRepository(Protocol):
    def append_message(self, session_id: str, message: ChatMessage) -> None:
        ...

    def list_messages(self, session_id: str, *, limit: int = 100) -> list[ChatMessage]:
        """Return messages oldest-first within the limit (most recent `limit` turns)."""
        ...
