from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from evenor.adapters.persistence.orm import PendingReplyEditRow
from evenor.domain.models.pending_reply_edit import PendingReplyEdit


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

    def list_for_reply(
        self, tenant_id: int, pending_reply_id: int, *, limit: int = 100
    ) -> list[PendingReplyEdit]:
        rows = self._session.scalars(
            select(PendingReplyEditRow)
            .where(
                PendingReplyEditRow.tenant_id == tenant_id,
                PendingReplyEditRow.pending_reply_id == pending_reply_id,
            )
            .order_by(PendingReplyEditRow.created_at.desc())
            .limit(limit)
        ).all()
        return [_row_to_edit(row) for row in rows]

    def list_for_tenant(self, tenant_id: int, *, limit: int = 50) -> list[PendingReplyEdit]:
        rows = self._session.scalars(
            select(PendingReplyEditRow)
            .where(PendingReplyEditRow.tenant_id == tenant_id)
            .order_by(PendingReplyEditRow.created_at.desc())
            .limit(limit)
        ).all()
        return [_row_to_edit(row) for row in rows]
