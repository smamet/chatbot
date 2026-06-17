from __future__ import annotations

import httpx

from chatbot.adapters.mail.smtp_sender import EmailSendError
from chatbot.adapters.mail.types import EmailMessage

_MAILJET_URL = "https://api.mailjet.com/v3.1/send"
_MAILJET_USER_URL = "https://api.mailjet.com/v3/REST/user"


class MailjetEmailSender:
    def __init__(self, *, api_key: str, api_secret: str) -> None:
        self._api_key = api_key
        self._api_secret = api_secret

    def verify_connection(self) -> None:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    _MAILJET_USER_URL,
                    auth=(self._api_key, self._api_secret),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmailSendError(f"Mailjet connection failed: {exc}") from exc

    def send(self, message: EmailMessage) -> None:
        msg_payload: dict = {
            "From": {"Email": message.from_addr},
            "To": [{"Email": message.to_addr}],
            "Subject": message.subject,
            "TextPart": message.body_text,
        }
        if message.body_html:
            msg_payload["HTMLPart"] = message.body_html
        payload = {"Messages": [msg_payload]}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    _MAILJET_URL,
                    json=payload,
                    auth=(self._api_key, self._api_secret),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmailSendError(f"Mailjet send failed: {exc}") from exc
