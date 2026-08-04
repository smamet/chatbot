from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class MailConnectionProvider(StrEnum):
    MICROSOFT_OAUTH = "microsoft_oauth"
    GOOGLE_OAUTH = "google_oauth"


@dataclass(frozen=True)
class MailConnection:
    id: int
    tenant_id: int
    label: str
    provider: MailConnectionProvider
    mailbox_email: str
    config: dict[str, Any]
    active: bool
    created_at: datetime
    updated_at: datetime
