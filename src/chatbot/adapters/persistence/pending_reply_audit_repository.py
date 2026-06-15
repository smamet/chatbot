from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import PendingReplyAuditEventRow
from chatbot.domain.models.pending_reply_audit import PendingReplyAuditEvent, ValidationAuditAction


def _row_to_event(row: PendingReplyAuditEventRow) -> PendingReplyAuditEvent:
    created = row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at
    try:
        action = ValidationAuditAction(row.action)
    except ValueError:
        action = ValidationAuditAction.APPROVED
    return PendingReplyAuditEvent(
        id=row.id,
        tenant_id=row.tenant_id,
        pending_reply_id=row.pending_reply_id,
        action=action,
        actor_email=row.actor_email,
        detail_json=row.detail_json,
        created_at=created,
    )


class SqlAlchemyPendingReplyAuditRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        tenant_id: int,
        pending_reply_id: int,
        action: ValidationAuditAction,
        actor_email: str,
        detail: dict | None = None,
    ) -> PendingReplyAuditEvent:
        row = PendingReplyAuditEventRow(
            tenant_id=tenant_id,
            pending_reply_id=pending_reply_id,
            action=action.value,
            actor_email=actor_email,
            detail_json=json.dumps(detail, ensure_ascii=False) if detail else None,
            created_at=datetime.now(UTC),
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_event(row)

    def list_for_tenant(self, tenant_id: int, *, limit: int = 50) -> list[PendingReplyAuditEvent]:
        rows = self._session.scalars(
            select(PendingReplyAuditEventRow)
            .where(PendingReplyAuditEventRow.tenant_id == tenant_id)
            .order_by(PendingReplyAuditEventRow.created_at.desc())
            .limit(limit)
        ).all()
        return [_row_to_event(row) for row in rows]

    def list_for_reply(
        self, tenant_id: int, pending_reply_id: int, *, limit: int = 100
    ) -> list[PendingReplyAuditEvent]:
        rows = self._session.scalars(
            select(PendingReplyAuditEventRow)
            .where(
                PendingReplyAuditEventRow.tenant_id == tenant_id,
                PendingReplyAuditEventRow.pending_reply_id == pending_reply_id,
            )
            .order_by(PendingReplyAuditEventRow.created_at.desc())
            .limit(limit)
        ).all()
        return [_row_to_event(row) for row in rows]
