from __future__ import annotations

from chatbot.adapters.mail.mailgun_sender import MailgunEmailSender
from chatbot.adapters.mail.mailjet_sender import MailjetEmailSender
from chatbot.adapters.mail.protocol import EmailSender
from chatbot.adapters.mail.smtp_sender import EmailSendError, SmtpEmailSender, _parse_use_tls
from chatbot.domain.models.connector_schema import EmailOutboundProvider, resolve_email_outbound_provider


def _require(config: dict, key: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise EmailSendError(f"Missing email config: {key}")
    return value


def build_email_sender(config: dict) -> EmailSender:
    provider = resolve_email_outbound_provider(config)
    if provider == EmailOutboundProvider.SMTP.value:
        host = _require(config, "smtp_host")
        port_raw = str(config.get("smtp_port", "587")).strip() or "587"
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise EmailSendError("Invalid smtp_port") from exc
        access_token = str(config.get("_resolved_access_token", "")).strip() or None
        return SmtpEmailSender(
            host=host,
            port=port,
            username=str(config.get("smtp_username", "")).strip(),
            password=str(config.get("smtp_password", "")).strip(),
            use_tls=_parse_use_tls(config.get("smtp_use_tls"), default=True),
            access_token=access_token,
        )
    if provider == EmailOutboundProvider.MAILJET.value:
        return MailjetEmailSender(
            api_key=_require(config, "mailjet_api_key"),
            api_secret=_require(config, "mailjet_api_secret"),
        )
    if provider == EmailOutboundProvider.MAILGUN.value:
        region = str(config.get("mailgun_region", "us")).strip().lower() or "us"
        return MailgunEmailSender(
            api_key=_require(config, "mailgun_api_key"),
            domain=_require(config, "mailgun_domain"),
            region=region,
        )
    raise EmailSendError(f"Unsupported outbound_provider: {provider}")
