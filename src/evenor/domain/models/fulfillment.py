from __future__ import annotations

from enum import StrEnum


class PendingReplyStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class FulfillmentKind(StrEnum):
    REPLY_ONLY = "reply_only"
    ERPNEXT_QUOTE = "erpnext_quote"
