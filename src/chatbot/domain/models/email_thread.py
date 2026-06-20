from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class EmailThread:
    id: int
    tenant_id: int
    from_addr: str
    thread_key: str
    root_message_id: str | None
    normalized_subject: str
    last_activity_at: datetime
    created_at: datetime
