from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import OutboundEmailMessageRow
from chatbot.domain.models.outbound_email_message import OutboundEmailMessage


def _row_to_outbound(row: OutboundEmailMessageRow) -> OutboundEmailMessage:
    sent = row.sent_at
    if sent.tzinfo is None:
        sent = sent.replace(tzinfo=UTC)
    return OutboundEmailMessage(
        id=row.id,
        tenant_id=row.tenant_id,
        thread_id=row.thread_id,
        message_id=row.message_id,
        in_reply_to=row.in_reply_to,
        references_header=row.references_header,
        pending_reply_id=row.pending_reply_id,
        sent_at=sent,
    )


class SqlAlchemyOutboundEmailMessageRepository:
    def __init__(self, session: Session, *, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def find_by_message_id(self, message_id: str) -> OutboundEmailMessage | None:
        mid = message_id.strip()
        if not mid:
            return None
        row = self._session.scalar(
            select(OutboundEmailMessageRow).where(
                OutboundEmailMessageRow.tenant_id == self._tenant_id,
                OutboundEmailMessageRow.message_id == mid,
            )
        )
        return _row_to_outbound(row) if row else None

    def find_latest_for_thread(self, thread_id: int) -> OutboundEmailMessage | None:
        row = self._session.scalar(
            select(OutboundEmailMessageRow)
            .where(
                OutboundEmailMessageRow.tenant_id == self._tenant_id,
                OutboundEmailMessageRow.thread_id == thread_id,
            )
            .order_by(OutboundEmailMessageRow.id.desc())
            .limit(1)
        )
        return _row_to_outbound(row) if row else None

    def record(
        self,
        *,
        thread_id: int,
        message_id: str,
        in_reply_to: str | None,
        references_header: str | None,
        pending_reply_id: int | None,
        sent_at: datetime | None = None,
    ) -> OutboundEmailMessage:
        row = OutboundEmailMessageRow(
            tenant_id=self._tenant_id,
            thread_id=thread_id,
            message_id=message_id.strip(),
            in_reply_to=in_reply_to,
            references_header=references_header,
            pending_reply_id=pending_reply_id,
            sent_at=sent_at or datetime.now(UTC),
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_outbound(row)
