from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import EmailThreadRow
from chatbot.domain.models.email_thread import EmailThread


def _row_to_thread(row: EmailThreadRow) -> EmailThread:
    last = row.last_activity_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return EmailThread(
        id=row.id,
        tenant_id=row.tenant_id,
        from_addr=row.from_addr,
        thread_key=row.thread_key,
        root_message_id=row.root_message_id,
        normalized_subject=row.normalized_subject,
        last_activity_at=last,
        created_at=created,
    )


class SqlAlchemyEmailThreadRepository:
    def __init__(self, session: Session, *, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def find_by_id(self, thread_id: int) -> EmailThread | None:
        row = self._session.get(EmailThreadRow, thread_id)
        if row is None or row.tenant_id != self._tenant_id:
            return None
        return _row_to_thread(row)

    def find_by_key(self, from_addr: str, thread_key: str) -> EmailThread | None:
        row = self._session.scalar(
            select(EmailThreadRow).where(
                EmailThreadRow.tenant_id == self._tenant_id,
                EmailThreadRow.from_addr == from_addr.strip().lower(),
                EmailThreadRow.thread_key == thread_key.strip(),
            )
        )
        return _row_to_thread(row) if row else None

    def list_open_by_sender(self, from_addr: str, *, since: datetime) -> list[EmailThread]:
        rows = self._session.scalars(
            select(EmailThreadRow)
            .where(
                EmailThreadRow.tenant_id == self._tenant_id,
                EmailThreadRow.from_addr == from_addr.strip().lower(),
                EmailThreadRow.last_activity_at >= since,
            )
            .order_by(EmailThreadRow.last_activity_at.desc())
        ).all()
        return [_row_to_thread(row) for row in rows]

    def create(
        self,
        *,
        from_addr: str,
        thread_key: str,
        root_message_id: str | None,
        normalized_subject: str,
        last_activity_at: datetime,
    ) -> EmailThread:
        now = datetime.now(UTC)
        row = EmailThreadRow(
            tenant_id=self._tenant_id,
            from_addr=from_addr.strip().lower(),
            thread_key=thread_key.strip(),
            root_message_id=root_message_id,
            normalized_subject=normalized_subject,
            last_activity_at=last_activity_at,
            created_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_thread(row)

    def touch_activity(self, thread_id: int, at: datetime) -> None:
        row = self._session.get(EmailThreadRow, thread_id)
        if row is None or row.tenant_id != self._tenant_id:
            return
        row.last_activity_at = at
        self._session.flush()
