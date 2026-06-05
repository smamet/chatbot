from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


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
