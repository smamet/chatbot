from __future__ import annotations

from sqlalchemy.orm import Session

from evenor.application.connector_service import ConnectorService
from evenor.application.mail_connection_service import MailConnectionService
from evenor.application.mail_oauth_service import prepare_oauth_mail_config
from evenor.config.settings import Settings, get_settings
from evenor.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from evenor.domain.models.connector_schema import is_oauth_auth_type, resolve_email_auth_type, runtime_mail_config_keys


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
    force_oauth_refresh: bool = False,
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
            force_oauth_refresh=force_oauth_refresh,
        )
    if not is_oauth_auth_type(resolve_email_auth_type(config)):
        return config
    mail_cfg, updated = prepare_oauth_mail_config(
        config,
        direction=direction.value,
        settings=resolved_settings,
        force_refresh=force_oauth_refresh,
    )
    if updated is None:
        return mail_cfg
    persisted = {key: value for key, value in updated.items() if key not in runtime_mail_config_keys()}
    svc = ConnectorService(session)
    svc.upsert(
        tenant_id=connector.tenant_id,
        direction=direction,
        type=ConnectorType.EMAIL,
        mode=connector.mode,
        config=persisted,
        active=connector.active,
    )
    session.flush()
    return mail_cfg
