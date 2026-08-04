from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class OutboundEmailMessage:
    id: int
    tenant_id: int
    thread_id: int
    message_id: str
    in_reply_to: str | None
    references_header: str | None
    pending_reply_id: int | None
    sent_at: datetime
