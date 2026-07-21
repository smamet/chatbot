from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class HookStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class HookEvent:
    id: int
    tenant_id: int
    session_id: str
    type: str
    payload_json: str
    status: HookStatus
    attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None
