from __future__ import annotations

import smtplib
from email.message import EmailMessage as StdEmailMessage

from chatbot.adapters.mail.types import EmailMessage


class EmailSendError(RuntimeError):
    pass


class SmtpEmailSender:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password

    def send(self, message: EmailMessage) -> None:
        msg = StdEmailMessage()
        msg["From"] = message.from_addr
        msg["To"] = message.to_addr
        msg["Subject"] = message.subject
        msg.set_content(message.body_text)
        try:
            with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                smtp.starttls()
                if self._username:
                    smtp.login(self._username, self._password)
                smtp.send_message(msg)
        except smtplib.SMTPException as exc:
            raise EmailSendError(f"SMTP send failed: {exc}") from exc
