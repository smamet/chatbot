from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmailAttachment:
    filename: str
    data: bytes
    mime_type: str = "application/pdf"


@dataclass(frozen=True)
class EmailMessage:
    to_addr: str
    subject: str
    body_text: str
    from_addr: str
    attachments: tuple[EmailAttachment, ...] = field(default_factory=tuple)
