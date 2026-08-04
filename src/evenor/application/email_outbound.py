from __future__ import annotations

import re

from evenor.adapters.mail.body_format import format_email_bodies
from evenor.adapters.mail.factory import build_email_sender
from evenor.adapters.mail.smtp_sender import EmailSendError
from evenor.adapters.mail.types import EmailAttachment, EmailMessage
from evenor.application.email_message_id import generate_message_id
from evenor.application.email_threading import EmailThreadingContext
from evenor.domain.models.outbound_attachment import OutboundAttachment

_RE_PREFIX = re.compile(r"^re:\s*", re.IGNORECASE)


def coalesce_stored_email_subject(
    *,
    stored_draft_subject: str | None,
    connector_config: dict | None = None,
    inbound_subject: str | None = None,
) -> str | None:
    """Ignore auto-generated placeholders so inbound subject can apply."""
    stored = str(stored_draft_subject or "").strip()
    if not stored:
        return None
    generic = resolve_email_subject(connector_config=connector_config, inbound_subject=None)
    if stored == generic:
        return None
    inbound = str(inbound_subject or "").strip()
    if inbound and stored.lower() == inbound.lower():
        return None
    return stored


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
    threading: EmailThreadingContext | None = None,
) -> str:
    to = to_addr.strip()
    if not to:
        raise EmailSendError("Recipient email address is empty")
    from_addr = str(config.get("from_addr", "")).strip()
    if not from_addr:
        raise EmailSendError("Missing from_addr in email connector config")
    draft = subject.strip() if subject and subject.strip() else None
    resolved_subject = resolve_email_subject(
        draft_subject=coalesce_stored_email_subject(
            stored_draft_subject=draft,
            connector_config=config,
        ),
        connector_config=config,
    )
    email_attachments = tuple(
        EmailAttachment(filename=a.filename, data=a.data, mime_type=a.mime_type)
        for a in (attachments or [])
    )
    body_text, rendered_html = format_email_bodies(body, html_fragment=body_html)
    message_id = generate_message_id(from_addr)
    sender = build_email_sender(config)
    return sender.send(
        EmailMessage(
            to_addr=to,
            subject=resolved_subject,
            body_text=body_text,
            body_html=rendered_html,
            from_addr=from_addr,
            attachments=email_attachments,
            message_id=message_id,
            in_reply_to=threading.in_reply_to if threading else None,
            references=threading.references if threading else None,
        )
    )
