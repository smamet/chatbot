from __future__ import annotations

import smtplib
from email.message import EmailMessage as StdEmailMessage

from chatbot.adapters.mail.types import EmailMessage


class EmailSendError(RuntimeError):
    pass


def _parse_use_tls(config_value: object, *, default: bool = True) -> bool:
    if config_value is None or config_value == "":
        return default
    if isinstance(config_value, bool):
        return config_value
    return str(config_value).strip().lower() in ("1", "true", "on", "yes")


class SmtpEmailSender:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        use_tls: bool = True,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_tls = use_tls

    def send(self, message: EmailMessage) -> None:
        msg = StdEmailMessage()
        msg["From"] = message.from_addr
        msg["To"] = message.to_addr
        msg["Subject"] = message.subject
        if message.attachments:
            msg.set_content(message.body_text)
            for att in message.attachments:
                msg.add_attachment(
                    att.data,
                    maintype="application",
                    subtype=att.mime_type.split("/")[-1] if "/" in att.mime_type else "octet-stream",
                    filename=att.filename,
                )
        else:
            msg.set_content(message.body_text)
        try:
            with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                if self._use_tls:
                    smtp.starttls()
                if self._username:
                    smtp.login(self._username, self._password)
                smtp.send_message(msg)
        except smtplib.SMTPException as exc:
            raise EmailSendError(f"SMTP send failed: {exc}") from exc
