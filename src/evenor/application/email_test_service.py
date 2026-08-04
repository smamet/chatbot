from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from evenor.adapters.mail.factory import build_email_sender
from evenor.adapters.mail.smtp_sender import EmailSendError
from evenor.adapters.mail.types import EmailMessage
from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.application.connector_service import ConnectorService
from evenor.config.settings import Settings
from evenor.domain.models.connector import ConnectorDirection, ConnectorType
from evenor.domain.models.tenant import Tenant
from evenor.mail.listener import run_once_for_tenant


class EmailTestError(RuntimeError):
    pass


def build_dev_inject_smtp_config(settings: Settings) -> dict:
    """SMTP config for injecting inbound test mail into GreenMail (not the OUT connector)."""
    return {
        "outbound_provider": "smtp",
        "smtp_host": settings.dev_mail_inject_smtp_host,
        "smtp_port": str(settings.dev_mail_inject_smtp_port),
        "smtp_username": "",
        "smtp_password": "",
        "smtp_use_tls": False,
    }


def _mailbox_address(config_in: dict) -> str:
    addr = str(config_in.get("username", "")).strip()
    if not addr:
        raise EmailTestError("Email inbound connector has no mailbox username")
    return addr


def inject_test_email(
    settings: Settings,
    config_in: dict,
    *,
    from_addr: str,
    subject: str,
    body: str,
) -> None:
    sender_addr = from_addr.strip().lower()
    if not sender_addr:
        raise EmailTestError("From address is required")
    if not body.strip():
        raise EmailTestError("Body is required")
    to_addr = _mailbox_address(config_in)
    resolved_subject = (subject or "Test email").strip()
    sender = build_email_sender(build_dev_inject_smtp_config(settings))
    sender.send(
        EmailMessage(
            to_addr=to_addr,
            subject=resolved_subject,
            body_text=body.strip(),
            from_addr=sender_addr,
        )
    )


def poll_tenant_now(
    session_factory: sessionmaker[Session],
    settings: Settings,
    tenant: Tenant,
) -> int:
    if not tenant.active:
        raise EmailTestError("Tenant is inactive")
    return run_once_for_tenant(session_factory, settings, tenant_id=tenant.id)


def get_email_test_connectors(session: Session, tenant_id: int) -> dict:
    connectors = ConnectorService(SqlAlchemyConnectorRepository(session))
    config_in = connectors.get_email_config(tenant_id, outbound=False)
    if not config_in:
        raise EmailTestError("No active email inbound connector")
    return config_in
