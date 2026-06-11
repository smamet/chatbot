from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import PendingReplyRow
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus


def _row_to_pending(row: PendingReplyRow) -> PendingReply:
    created = row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at
    updated = row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at
    try:
        fulfillment_kind = FulfillmentKind(row.fulfillment_kind)
    except ValueError:
        fulfillment_kind = FulfillmentKind.REPLY_ONLY
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
        hook_event_id=row.hook_event_id,
        fulfillment_kind=fulfillment_kind,
        quote_proposal_json=row.quote_proposal_json,
        quote_resolved_json=row.quote_resolved_json,
        quote_external_id=row.quote_external_id,
        attachments_json=row.attachments_json,
        fulfillment_error=row.fulfillment_error,
        draft_html=row.draft_html,
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
        draft_html: str | None = None,
        hook_event_id: int | None = None,
        fulfillment_kind: FulfillmentKind = FulfillmentKind.REPLY_ONLY,
        quote_proposal_json: str | None = None,
        quote_resolved_json: str | None = None,
        quote_external_id: str | None = None,
        attachments_json: str | None = None,
    ) -> PendingReply:
        now = datetime.now(UTC)
        row = PendingReplyRow(
            tenant_id=tenant_id,
            connector_id=connector_id,
            session_id=session_id,
            channel=channel,
            recipient_id=recipient_id,
            draft_text=draft_text,
            draft_html=draft_html,
            status=PendingReplyStatus.PENDING.value,
            hook_event_id=hook_event_id,
            fulfillment_kind=fulfillment_kind.value,
            quote_proposal_json=quote_proposal_json,
            quote_resolved_json=quote_resolved_json,
            quote_external_id=quote_external_id,
            attachments_json=attachments_json,
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

    def update_quote_fields(
        self,
        reply_id: int,
        *,
        quote_resolved_json: str | None = None,
        quote_external_id: str | None = None,
        attachments_json: str | None = None,
        clear_attachments_json: bool = False,
        fulfillment_error: str | None = None,
    ) -> PendingReply | None:
        row = self._session.get(PendingReplyRow, reply_id)
        if row is None:
            return None
        if quote_resolved_json is not None:
            row.quote_resolved_json = quote_resolved_json
        if quote_external_id is not None:
            row.quote_external_id = quote_external_id
        if clear_attachments_json:
            row.attachments_json = None
        elif attachments_json is not None:
            row.attachments_json = attachments_json
        if fulfillment_error is not None:
            row.fulfillment_error = fulfillment_error
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_pending(row)

    def update_draft(
        self,
        reply_id: int,
        *,
        draft_text: str | None = None,
        draft_html: str | None = None,
    ) -> PendingReply | None:
        row = self._session.get(PendingReplyRow, reply_id)
        if row is None:
            return None
        if draft_text is not None:
            row.draft_text = draft_text
        if draft_html is not None:
            row.draft_html = draft_html
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_pending(row)
