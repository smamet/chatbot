from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import MessageRow
from chatbot.application.context_debug import context_debug_from_json, context_debug_to_json
from chatbot.domain.models.message import ChatMessage, MessageRole


class SqlAlchemyConversationRepository:
    def __init__(self, session: Session, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        row = MessageRow(
            tenant_id=self._tenant_id,
            session_id=session_id,
            role=message.role.value,
            content=message.content,
            context_debug_json=context_debug_to_json(message.context_debug),
        )
        self._session.add(row)
        self._session.flush()

    def list_messages(self, session_id: str, *, limit: int = 100) -> list[ChatMessage]:
        stmt = (
            select(MessageRow)
            .where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
            )
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
            out.append(
                ChatMessage(
                    role=role,
                    content=r.content,
                    context_debug=context_debug_from_json(r.context_debug_json),
                )
            )
        return out

    def last_user_message_before(self, session_id: str, before: datetime) -> str | None:
        stmt = (
            select(MessageRow.content)
            .where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
                MessageRow.role == MessageRole.USER.value,
                MessageRow.created_at <= before,
            )
            .order_by(desc(MessageRow.id))
            .limit(1)
        )
        return self._session.scalar(stmt)

    def list_session_ids(self, *, limit: int = 100) -> list[str]:
        stmt = (
            select(MessageRow.session_id)
            .where(MessageRow.tenant_id == self._tenant_id)
            .group_by(MessageRow.session_id)
            .order_by(desc(func.max(MessageRow.id)))
            .limit(limit)
        )
        return list(self._session.scalars(stmt))

    def clear_session(self, session_id: str) -> int:
        result = self._session.execute(
            delete(MessageRow).where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
            )
        )
        self._session.flush()
        return int(result.rowcount or 0)
