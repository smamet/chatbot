from __future__ import annotations

from chatbot.adapters.mail.factory import build_email_sender
from chatbot.adapters.mail.smtp_sender import EmailSendError
from chatbot.adapters.mail.types import EmailMessage


def send_email_reply(
    *,
    config: dict,
    to_addr: str,
    body: str,
    subject: str | None = None,
) -> None:
    to = to_addr.strip()
    if not to:
        raise EmailSendError("Recipient email address is empty")
    from_addr = str(config.get("from_addr", "")).strip()
    if not from_addr:
        raise EmailSendError("Missing from_addr in email connector config")
    resolved_subject = (subject or str(config.get("default_subject", "")).strip() or "Reply").strip()
    sender = build_email_sender(config)
    sender.send(
        EmailMessage(
            to_addr=to,
            subject=resolved_subject,
            body_text=body,
            from_addr=from_addr,
        )
    )
