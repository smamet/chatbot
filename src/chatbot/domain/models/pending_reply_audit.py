from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ValidationAuditAction(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ATTACHMENT_ADDED = "attachment_added"
    ATTACHMENT_REMOVED = "attachment_removed"
    RESOLVE_PRODUCTS = "resolve_products"
    REFRESH_PDF = "refresh_pdf"


@dataclass(frozen=True)
class PendingReplyAuditEvent:
    id: int
    tenant_id: int
    pending_reply_id: int
    action: ValidationAuditAction
    actor_email: str
    detail_json: str | None
    created_at: datetime


@dataclass(frozen=True)
class ValidationTimelineEntry:
    """Unified view for activity feed (edits + audit events)."""

    kind: str  # "edit" | "audit"
    pending_reply_id: int
    actor_email: str
    created_at: datetime
    action: str  # edit | approved | rejected | ...
    summary: str
    diff: str | None = None
    detail_json: str | None = None
