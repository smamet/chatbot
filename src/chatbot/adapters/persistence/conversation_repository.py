from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import MessageRow
from chatbot.application.context_debug import context_debug_from_json, context_debug_to_json
from chatbot.domain.models.message import ChatMessage, MessageRole


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _row_to_message(row: MessageRow) -> ChatMessage:
    try:
        role = MessageRole(row.role)
    except ValueError:
        role = MessageRole.USER
    return ChatMessage(
        role=role,
        content=row.content,
        context_debug=context_debug_from_json(row.context_debug_json),
        created_at=_as_utc(row.created_at),
    )


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
        return [_row_to_message(r) for r in rows]

    def list_messages_before(
        self, session_id: str, before: datetime, *, limit: int = 200
    ) -> list[ChatMessage]:
        stmt = (
            select(MessageRow)
            .where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
                MessageRow.created_at <= before,
            )
            .order_by(MessageRow.id)
            .limit(limit)
        )
        rows = list(self._session.scalars(stmt))
        return [_row_to_message(r) for r in rows]

    def last_user_message_before(self, session_id: str, before: datetime) -> str | None:
        meta = self.last_user_message_with_time_before(session_id, before)
        return meta[0] if meta else None

    def last_user_message_with_time_before(
        self, session_id: str, before: datetime
    ) -> tuple[str, datetime] | None:
        stmt = (
            select(MessageRow.content, MessageRow.created_at)
            .where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
                MessageRow.role == MessageRole.USER.value,
                MessageRow.created_at <= before,
            )
            .order_by(desc(MessageRow.id))
            .limit(1)
        )
        row = self._session.execute(stmt).first()
        if row is None:
            return None
        return row[0], _as_utc(row[1])

    def list_session_ids(self, *, limit: int = 100) -> list[str]:
        stmt = (
            select(MessageRow.session_id)
            .where(MessageRow.tenant_id == self._tenant_id)
            .group_by(MessageRow.session_id)
            .order_by(desc(func.max(MessageRow.id)))
            .limit(limit)
        )
        return list(self._session.scalars(stmt))

    def update_user_message_content(
        self,
        session_id: str,
        *,
        old_content: str,
        new_content: str,
        before: datetime | None = None,
    ) -> bool:
        stmt = (
            select(MessageRow)
            .where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
                MessageRow.role == MessageRole.USER.value,
                MessageRow.content == old_content,
            )
            .order_by(desc(MessageRow.id))
            .limit(1)
        )
        if before is not None:
            stmt = stmt.where(MessageRow.created_at <= before)
        row = self._session.scalar(stmt)
        if row is None:
            return False
        row.content = new_content
        self._session.flush()
        return True

    def update_assistant_message_content(
        self,
        session_id: str,
        *,
        old_content: str,
        new_content: str,
    ) -> bool:
        stmt = (
            select(MessageRow)
            .where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
                MessageRow.role == MessageRole.ASSISTANT.value,
                MessageRow.content == old_content,
            )
            .order_by(desc(MessageRow.id))
            .limit(1)
        )
        row = self._session.scalar(stmt)
        if row is None:
            return False
        row.content = new_content
        self._session.flush()
        return True

    def clear_session(self, session_id: str) -> int:
        result = self._session.execute(
            delete(MessageRow).where(
                MessageRow.tenant_id == self._tenant_id,
                MessageRow.session_id == session_id,
            )
        )
        self._session.flush()
        return int(result.rowcount or 0)
