from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import MessageRow
from chatbot.domain.models.message import ChatMessage, MessageRole


class SqlAlchemyConversationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        row = MessageRow(
            session_id=session_id,
            role=message.role.value,
            content=message.content,
        )
        self._session.add(row)
        self._session.flush()

    def list_messages(self, session_id: str, *, limit: int = 100) -> list[ChatMessage]:
        stmt = (
            select(MessageRow)
            .where(MessageRow.session_id == session_id)
            .order_by(desc(MessageRow.id))
            .limit(limit)
        )
        rows = list(self._session.scalars(stmt))
        rows.reverse()
        out: list[ChatMessage] = []
        for r in rows:
            try:
                role = MessageRole(r.role)
            except ValueError:
                role = MessageRole.USER
            out.append(ChatMessage(role=role, content=r.content))
        return out
