from __future__ import annotations

import httpx

from chatbot.adapters.mail.smtp_sender import EmailSendError
from chatbot.adapters.mail.types import EmailMessage

_MAILGUN_BASE = {
    "us": "https://api.mailgun.net/v3",
    "eu": "https://api.eu.mailgun.net/v3",
}


class MailgunEmailSender:
    def __init__(self, *, api_key: str, domain: str, region: str = "us") -> None:
        self._api_key = api_key
        self._domain = domain
        base = _MAILGUN_BASE.get(region, _MAILGUN_BASE["us"])
        self._url = f"{base}/{domain}/messages"

    def send(self, message: EmailMessage) -> None:
        data = {
            "from": message.from_addr,
            "to": message.to_addr,
            "subject": message.subject,
            "text": message.body_text,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self._url,
                    data=data,
                    auth=("api", self._api_key),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmailSendError(f"Mailgun send failed: {exc}") from exc
