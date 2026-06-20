from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MailDraftStatus(StrEnum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MailDraft:
    id: int
    tenant_id: int
    imap_uid: str
    from_addr: str
    to_addr: str
    subject: str
    body_in: str
    draft_reply: str
    status: MailDraftStatus
    created_at: datetime
    updated_at: datetime
    body_new: str = ""
    thread_id: int | None = None
    message_id: str = ""
    in_reply_to: str = ""
    references_header: str | None = None
    normalized_subject: str = ""
    thread_resolution_json: str | None = None
