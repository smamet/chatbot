from __future__ import annotations

import smtplib
from email.message import EmailMessage as StdEmailMessage

from chatbot.adapters.mail.types import EmailMessage
from chatbot.adapters.mail.xoauth2 import build_xoauth2_string


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
        access_token: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_tls = use_tls
        self._access_token = (access_token or "").strip() or None

    def _xoauth2_response(self, _challenge: bytes = b"") -> str:
        if not self._username or not self._access_token:
            raise EmailSendError("SMTP username and OAuth access token are required")
        return build_xoauth2_string(self._username, self._access_token)

    def _authenticate(self, smtp: smtplib.SMTP) -> None:
        if self._access_token:
            if not self._username:
                raise EmailSendError("SMTP username is required for OAuth")
            smtp.auth("XOAUTH2", self._xoauth2_response)
            return
        if self._username:
            smtp.login(self._username, self._password)

    def _smtp_envelope_from(self, message: EmailMessage) -> str:
        """MAIL FROM used in SMTP envelope. For OAuth must be the authenticated mailbox."""
        if self._access_token and self._username:
            return self._username
        return message.from_addr

    def _prepare_connection(self, smtp: smtplib.SMTP) -> None:
        if self._use_tls:
            smtp.starttls()
            smtp.ehlo()
        self._authenticate(smtp)

    def verify_connection(self) -> None:
        try:
            with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                self._prepare_connection(smtp)
                smtp.noop()
        except smtplib.SMTPException as exc:
            raise EmailSendError(f"SMTP connection failed: {exc}") from exc

    def send(self, message: EmailMessage) -> None:
        msg = StdEmailMessage()
        msg["From"] = message.from_addr
        msg["To"] = message.to_addr
        msg["Subject"] = message.subject
        msg.set_content(message.body_text)
        if message.body_html:
            msg.add_alternative(message.body_html, subtype="html")
        for att in message.attachments:
            msg.add_attachment(
                att.data,
                maintype="application",
                subtype=att.mime_type.split("/")[-1] if "/" in att.mime_type else "octet-stream",
                filename=att.filename,
            )
        envelope_from = self._smtp_envelope_from(message)
        try:
            with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                self._prepare_connection(smtp)
                smtp.sendmail(envelope_from, [message.to_addr], msg.as_bytes())
        except smtplib.SMTPSenderRefused as exc:
            raise EmailSendError(
                f"SMTP send refused by server (code {exc.smtp_code}). "
                "If using Microsoft OAuth, ensure SMTP AUTH is enabled on the mailbox "
                "and re-connect the Mail connection to obtain SMTP.Send scope."
            ) from exc
        except smtplib.SMTPException as exc:
            raise EmailSendError(f"SMTP send failed: {exc}") from exc
