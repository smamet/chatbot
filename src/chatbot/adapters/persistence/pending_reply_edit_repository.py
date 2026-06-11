from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import PendingReplyEditRow
from chatbot.domain.models.pending_reply_edit import PendingReplyEdit


def _row_to_edit(row: PendingReplyEditRow) -> PendingReplyEdit:
    created = row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at
    return PendingReplyEdit(
        id=row.id,
        tenant_id=row.tenant_id,
        pending_reply_id=row.pending_reply_id,
        edited_by=row.edited_by,
        body_before=row.body_before,
        body_after=row.body_after,
        diff=row.diff,
        created_at=created,
    )


class SqlAlchemyPendingReplyEditRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        tenant_id: int,
        pending_reply_id: int,
        edited_by: str,
        body_before: str,
        body_after: str,
        diff: str,
    ) -> PendingReplyEdit:
        row = PendingReplyEditRow(
            tenant_id=tenant_id,
            pending_reply_id=pending_reply_id,
            edited_by=edited_by,
            body_before=body_before,
            body_after=body_after,
            diff=diff,
            created_at=datetime.now(UTC),
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_edit(row)
