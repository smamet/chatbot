from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from chatbot.adapters.mail.factory import build_email_sender
from chatbot.adapters.mail.smtp_sender import EmailSendError
from chatbot.adapters.mail.types import EmailMessage
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import ConnectorDirection, ConnectorType
from chatbot.domain.models.tenant import Tenant
from chatbot.mail.listener import run_once_for_tenant


class EmailTestError(RuntimeError):
    pass


def _mailbox_address(config_in: dict, config_out: dict | None) -> str:
    addr = str(config_in.get("username", "")).strip()
    if addr:
        return addr
    if config_out:
        addr = str(config_out.get("from_addr", "")).strip()
    if not addr:
        raise EmailTestError("Email inbound connector has no mailbox username")
    return addr


def inject_test_email(
    config_in: dict,
    config_out: dict,
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
    to_addr = _mailbox_address(config_in, config_out)
    resolved_subject = (subject or "Test email").strip()
    sender = build_email_sender(config_out)
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


def get_email_test_connectors(session: Session, tenant_id: int) -> tuple[dict, dict]:
    connectors = ConnectorService(SqlAlchemyConnectorRepository(session))
    config_in = connectors.get_email_config(tenant_id, outbound=False)
    if not config_in:
        raise EmailTestError("No active email inbound connector")
    config_out = connectors.get_email_config(tenant_id, outbound=True)
    if not config_out:
        out = connectors.find(
            tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
        )
        config_out = out.config if out and out.active else None
    if not config_out:
        raise EmailTestError("No active email outbound connector for SMTP inject")
    return config_in, config_out
