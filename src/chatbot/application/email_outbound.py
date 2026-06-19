from __future__ import annotations

import re

from chatbot.adapters.mail.body_format import format_email_bodies
from chatbot.adapters.mail.factory import build_email_sender
from chatbot.adapters.mail.smtp_sender import EmailSendError
from chatbot.adapters.mail.types import EmailAttachment, EmailMessage
from chatbot.domain.models.outbound_attachment import OutboundAttachment

_RE_PREFIX = re.compile(r"^re:\s*", re.IGNORECASE)


def resolve_email_subject(
    *,
    draft_subject: str | None = None,
    connector_config: dict | None = None,
    inbound_subject: str | None = None,
) -> str:
    if draft_subject and str(draft_subject).strip():
        return str(draft_subject).strip()
    cfg = connector_config or {}
    default = str(cfg.get("default_subject", "")).strip()
    if default:
        return default
    inbound = str(inbound_subject or "").strip()
    if inbound:
        if _RE_PREFIX.match(inbound):
            return inbound
        return f"Re: {inbound}"
    return "Reply"


def send_email_reply(
    *,
    config: dict,
    to_addr: str,
    body: str,
    subject: str | None = None,
    body_html: str | None = None,
    attachments: list[OutboundAttachment] | None = None,
) -> None:
    to = to_addr.strip()
    if not to:
        raise EmailSendError("Recipient email address is empty")
    from_addr = str(config.get("from_addr", "")).strip()
    if not from_addr:
        raise EmailSendError("Missing from_addr in email connector config")
    draft = subject.strip() if subject and subject.strip() else None
    resolved_subject = resolve_email_subject(draft_subject=draft, connector_config=config)
    email_attachments = tuple(
        EmailAttachment(filename=a.filename, data=a.data, mime_type=a.mime_type)
        for a in (attachments or [])
    )
    body_text, rendered_html = format_email_bodies(body, html_fragment=body_html)
    sender = build_email_sender(config)
    sender.send(
        EmailMessage(
            to_addr=to,
            subject=resolved_subject,
            body_text=body_text,
            body_html=rendered_html,
            from_addr=from_addr,
            attachments=email_attachments,
        )
    )
