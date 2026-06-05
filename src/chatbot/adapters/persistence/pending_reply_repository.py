from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import PendingReplyRow
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus


def _row_to_pending(row: PendingReplyRow) -> PendingReply:
    created = row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at
    updated = row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at
    return PendingReply(
        id=row.id,
        tenant_id=row.tenant_id,
        connector_id=row.connector_id,
        session_id=row.session_id,
        channel=row.channel,
        recipient_id=row.recipient_id,
        draft_text=row.draft_text,
        status=PendingReplyStatus(row.status),
        created_at=created,
        updated_at=updated,
    )


class SqlAlchemyPendingReplyRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        tenant_id: int,
        connector_id: int,
        session_id: str,
        channel: str,
        recipient_id: str,
        draft_text: str,
    ) -> PendingReply:
        now = datetime.now(UTC)
        row = PendingReplyRow(
            tenant_id=tenant_id,
            connector_id=connector_id,
            session_id=session_id,
            channel=channel,
            recipient_id=recipient_id,
            draft_text=draft_text,
            status=PendingReplyStatus.PENDING.value,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_pending(row)

    def find_by_id(self, reply_id: int) -> PendingReply | None:
        row = self._session.get(PendingReplyRow, reply_id)
        return _row_to_pending(row) if row else None

    def list_pending(self, tenant_id: int, *, limit: int = 100) -> list[PendingReply]:
        rows = self._session.scalars(
            select(PendingReplyRow)
            .where(
                PendingReplyRow.tenant_id == tenant_id,
                PendingReplyRow.status == PendingReplyStatus.PENDING.value,
            )
            .order_by(PendingReplyRow.created_at.desc())
            .limit(limit)
        ).all()
        return [_row_to_pending(row) for row in rows]

    def count_pending(self, tenant_id: int) -> int:
        return len(self.list_pending(tenant_id, limit=10_000))

    def update_status(self, reply_id: int, status: PendingReplyStatus) -> PendingReply | None:
        row = self._session.get(PendingReplyRow, reply_id)
        if row is None:
            return None
        row.status = status.value
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_pending(row)
