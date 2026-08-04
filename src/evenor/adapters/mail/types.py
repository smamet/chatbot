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
    body_html: str | None = None
    attachments: tuple[EmailAttachment, ...] = field(default_factory=tuple)
    message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None
