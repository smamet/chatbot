from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from chatbot.adapters.mail.factory import build_email_sender
from chatbot.adapters.mail.imap_client import ImapError, ImapMailClient
from chatbot.adapters.mail.smtp_sender import EmailSendError, SmtpEmailSender, _parse_use_tls
from chatbot.application.mail_connection_service import MailConnectionService
from chatbot.application.mail_oauth_service import MailOAuthError, prepare_oauth_mail_config
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.models.connector_schema import (
    EmailAuthType,
    EmailOutboundProvider,
    is_oauth_auth_type,
    resolve_email_auth_type,
    resolve_email_outbound_provider,
)
from chatbot.domain.models.mail_connection import MailConnection


@dataclass(frozen=True, slots=True)
class ConnectorTestResult:
    ok: bool
    message: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_connector_connection_test(
    connector_type: str,
    direction: str,
    config: dict,
    *,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    if connector_type != "email":
        return ConnectorTestResult(
            ok=False,
            message="Connection test is only available for email connectors.",
            error="unsupported_connector",
        )
    try:
        if direction == "in":
            return _test_imap(config, session=session, tenant_id=tenant_id, settings=settings)
        if direction == "out":
            return _test_outbound(config, session=session, tenant_id=tenant_id, settings=settings)
        return ConnectorTestResult(ok=False, message="Invalid connector direction.", error="invalid_direction")
    except (ImapError, EmailSendError, MailOAuthError) as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))
    except Exception as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))


def run_mail_connection_test(
    connection: MailConnection,
    *,
    test: str,
    session,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    resolved_settings = settings or get_settings()
    svc = MailConnectionService(session)
    try:
        if test == "imap":
            mail_cfg, _updated = svc.resolve_runtime_config(
                connection, direction="in", settings=resolved_settings
            )
            return _test_imap(mail_cfg)
        if test == "smtp":
            mail_cfg, _updated = svc.resolve_runtime_config(
                connection, direction="out", settings=resolved_settings
            )
            return _test_outbound_smtp(mail_cfg)
        return ConnectorTestResult(ok=False, message="Invalid test type.", error="invalid_test")
    except (ImapError, EmailSendError, MailOAuthError) as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))
    except Exception as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))


def _test_imap(
    config: dict,
    *,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    mail_cfg = _mail_config_for_test(
        config, direction="in", session=session, tenant_id=tenant_id, settings=settings
    )
    client = ImapMailClient(mail_cfg)
    try:
        client.connect()
    finally:
        client.close()
    auth_label = "OAuth" if is_oauth_auth_type(resolve_email_auth_type(config)) else "password"
    return ConnectorTestResult(
        ok=True,
        message=f"IMAP connection OK ({auth_label}) — INBOX accessible.",
    )


def _test_outbound(
    config: dict,
    *,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    provider = resolve_email_outbound_provider(config)
    if provider == EmailOutboundProvider.SMTP.value:
        mail_cfg = _mail_config_for_test(
            config, direction="out", session=session, tenant_id=tenant_id, settings=settings
        )
        return _test_outbound_smtp(mail_cfg, source_config=config)
    sender = build_email_sender(config)
    sender.verify_connection()
    return ConnectorTestResult(ok=True, message=f"{provider.capitalize()} connection OK.")


def _test_outbound_smtp(mail_cfg: dict, *, source_config: dict | None = None) -> ConnectorTestResult:
    host = str(mail_cfg.get("smtp_host", "")).strip()
    if not host:
        raise EmailSendError("Missing smtp_host")
    port_raw = str(mail_cfg.get("smtp_port", "587")).strip() or "587"
    port = int(port_raw)
    access_token = str(mail_cfg.get("_resolved_access_token", "")).strip() or None
    SmtpEmailSender(
        host=host,
        port=port,
        username=str(mail_cfg.get("smtp_username", "")).strip(),
        password=str(mail_cfg.get("smtp_password", "")).strip(),
        use_tls=_parse_use_tls(mail_cfg.get("smtp_use_tls"), default=True),
        access_token=access_token,
    ).verify_connection()
    cfg = source_config or mail_cfg
    auth_label = "OAuth" if is_oauth_auth_type(resolve_email_auth_type(cfg)) else "password"
    return ConnectorTestResult(ok=True, message=f"SMTP connection OK ({auth_label}).")


def _mail_config_for_test(
    config: dict,
    *,
    direction: str,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> dict:
    resolved_settings = settings or get_settings()
    raw_id = config.get("mail_connection_id")
    if raw_id is not None and str(raw_id).strip() != "" and session is not None and tenant_id is not None:
        connection = MailConnectionService(session).get_for_tenant(int(raw_id), tenant_id)
        if connection is not None:
            mail_cfg, _updated = MailConnectionService(session).resolve_runtime_config(
                connection,
                direction=direction,
                settings=resolved_settings,
            )
            return mail_cfg
    auth_type = resolve_email_auth_type(config)
    if not is_oauth_auth_type(auth_type):
        return config
    mail_cfg, _updated = prepare_oauth_mail_config(
        config,
        direction=direction,
        settings=resolved_settings,
    )
    return mail_cfg
