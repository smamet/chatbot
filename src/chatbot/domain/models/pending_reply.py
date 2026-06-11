from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from chatbot.domain.models.fulfillment import FulfillmentKind


class PendingReplyStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PendingReply:
    id: int
    tenant_id: int
    connector_id: int
    session_id: str
    channel: str
    recipient_id: str
    draft_text: str
    status: PendingReplyStatus
    created_at: datetime
    updated_at: datetime
    hook_event_id: int | None = None
    fulfillment_kind: FulfillmentKind = FulfillmentKind.REPLY_ONLY
    quote_proposal_json: str | None = None
    quote_resolved_json: str | None = None
    quote_external_id: str | None = None
    attachments_json: str | None = None
    fulfillment_error: str | None = None
    draft_html: str | None = None
