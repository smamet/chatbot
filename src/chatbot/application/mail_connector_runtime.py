from __future__ import annotations

from sqlalchemy.orm import Session

from chatbot.application.connector_service import ConnectorService
from chatbot.application.mail_connection_service import MailConnectionService
from chatbot.application.mail_oauth_service import prepare_oauth_mail_config
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.connector_schema import is_oauth_auth_type, resolve_email_auth_type


def _connection_id_from_config(config: dict) -> int | None:
    raw = config.get("mail_connection_id")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def prepare_email_connector_config(
    connector: Connector,
    *,
    session: Session,
    direction: ConnectorDirection,
    settings: Settings | None = None,
) -> dict:
    """Resolve OAuth tokens and persist refreshed credentials when needed."""
    resolved_settings = settings or get_settings()
    config = dict(connector.config)
    connection_id = _connection_id_from_config(config)
    if connection_id is not None:
        return MailConnectionService(session).resolve_for_connector(
            connector,
            direction=direction.value,
            settings=resolved_settings,
        )
    if not is_oauth_auth_type(resolve_email_auth_type(config)):
        return config
    mail_cfg, updated = prepare_oauth_mail_config(
        config,
        direction=direction.value,
        settings=resolved_settings,
    )
    if updated is None:
        return mail_cfg
    svc = ConnectorService(session)
    svc.upsert(
        tenant_id=connector.tenant_id,
        direction=direction,
        type=ConnectorType.EMAIL,
        mode=connector.mode,
        config=updated,
        active=connector.active,
    )
    session.flush()
    return mail_cfg
