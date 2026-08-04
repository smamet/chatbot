from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PendingReplyEdit:
    id: int
    tenant_id: int
    pending_reply_id: int
    edited_by: str
    body_before: str
    body_after: str
    diff: str
    created_at: datetime
