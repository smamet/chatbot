from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from evenor.adapters.persistence.orm import HookEventRow
from evenor.domain.models.hook import HookEvent, HookStatus


def _row_to_hook(row: HookEventRow) -> HookEvent:
    return HookEvent(
        id=row.id,
        tenant_id=row.tenant_id,
        session_id=row.session_id,
        type=row.type,
        payload_json=row.payload_json,
        status=HookStatus(row.status),
        attempts=row.attempts,
        error=row.error,
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
        processed_at=(
            row.processed_at.replace(tzinfo=UTC)
            if row.processed_at and row.processed_at.tzinfo is None
            else row.processed_at
        ),
    )


class SqlAlchemyHookEventRepository:
    def __init__(self, session: Session, tenant_id: int | None = None) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def create(
        self,
        *,
        session_id: str,
        hook_type: str,
        payload_json: str,
        tenant_id: int | None = None,
    ) -> HookEvent:
        tid = tenant_id if tenant_id is not None else self._tenant_id
        if tid is None:
            raise ValueError("tenant_id required")
        now = datetime.now(UTC)
        row = HookEventRow(
            tenant_id=tid,
            session_id=session_id,
            type=hook_type,
            payload_json=payload_json,
            status=HookStatus.PENDING.value,
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_hook(row)

    def list_by_tenant(
        self, *, limit: int = 100, status: HookStatus | None = None, tenant_id: int | None = None
    ) -> list[HookEvent]:
        tid = tenant_id if tenant_id is not None else self._tenant_id
        stmt = select(HookEventRow)
        if tid is not None:
            stmt = stmt.where(HookEventRow.tenant_id == tid)
        if status is not None:
            stmt = stmt.where(HookEventRow.status == status.value)
        stmt = stmt.order_by(HookEventRow.id.desc()).limit(limit)
        return [_row_to_hook(r) for r in self._session.scalars(stmt)]

    def claim_pending(self, *, limit: int = 10) -> list[HookEvent]:
        stmt = (
            select(HookEventRow)
            .where(HookEventRow.status == HookStatus.PENDING.value)
            .order_by(HookEventRow.id.asc())
            .limit(limit)
        )
        rows = list(self._session.scalars(stmt))
        now = datetime.now(UTC)
        out: list[HookEvent] = []
        for row in rows:
            row.status = HookStatus.PROCESSING.value
            row.updated_at = now
            out.append(_row_to_hook(row))
        self._session.flush()
        return out

    def update_status(
        self,
        hook_id: int,
        *,
        status: HookStatus,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        now = datetime.now(UTC)
        row = self._session.get(HookEventRow, hook_id)
        if row is None:
            return
        row.status = status.value
        row.error = error
        row.updated_at = now
        if status in (HookStatus.DONE, HookStatus.FAILED):
            row.processed_at = now
        if increment_attempts:
            row.attempts += 1
        self._session.flush()

    def reset_to_pending(self, hook_id: int) -> None:
        self._session.execute(
            update(HookEventRow)
            .where(HookEventRow.id == hook_id)
            .values(status=HookStatus.PENDING.value, error=None, updated_at=datetime.now(UTC))
        )
        self._session.flush()
