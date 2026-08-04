from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.application.connector_service import ConnectorService
from evenor.application.mail_connection_service import (
    MailConnectionService,
    connector_auth_type_for_connection,
    strip_connector_oauth_fields,
)
from evenor.domain.models.connector import ConnectorDirection, ConnectorType
from evenor.domain.models.connector_schema import (
    EmailAuthType,
    is_oauth_auth_type,
    resolve_email_auth_type,
)
from evenor.domain.models.mail_connection import MailConnectionProvider


@dataclass(frozen=True, slots=True)
class MailConnectionMigrateResult:
    connections_created: int
    connectors_updated: int
    message: str


def _mailbox_from_connector(config: dict, *, direction: str) -> str:
    if direction == ConnectorDirection.IN.value:
        return str(config.get("username", "")).strip().lower()
    return str(config.get("smtp_username", "") or config.get("from_addr", "")).strip().lower()


def _client_id_for_provider(config: dict, provider: str) -> str:
    if provider == MailConnectionProvider.MICROSOFT_OAUTH.value:
        return str(config.get("microsoft_client_id", "")).strip()
    return str(config.get("google_client_id", "")).strip()


def _connection_config_from_connector(config: dict, provider: str) -> dict:
    if provider == MailConnectionProvider.MICROSOFT_OAUTH.value:
        keys = (
            "microsoft_client_id",
            "microsoft_client_secret",
            "oauth_refresh_token",
            "oauth_access_token",
            "oauth_token_expires_at",
        )
    else:
        keys = (
            "google_client_id",
            "google_client_secret",
            "oauth_refresh_token",
            "oauth_access_token",
            "oauth_token_expires_at",
        )
    return {key: config.get(key) for key in keys if config.get(key) is not None}


class MailConnectionMigrateService:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._connectors = SqlAlchemyConnectorRepository(session)
        self._conn_svc = MailConnectionService(session)

    def migrate_tenant(self, tenant_id: int) -> MailConnectionMigrateResult:
        email_connectors = [
            c
            for c in self._connectors.list_for_tenant(tenant_id)
            if c.type == ConnectorType.EMAIL and is_oauth_auth_type(resolve_email_auth_type(c.config))
        ]
        pending = [c for c in email_connectors if not c.config.get("mail_connection_id")]
        if not pending:
            return MailConnectionMigrateResult(
                connections_created=0,
                connectors_updated=0,
                message="No OAuth email connectors need migration.",
            )

        groups: dict[tuple[str, str, str], list] = {}
        for connector in pending:
            auth_type = resolve_email_auth_type(connector.config)
            mailbox = _mailbox_from_connector(connector.config, direction=connector.direction.value)
            client_id = _client_id_for_provider(connector.config, auth_type)
            if not mailbox or not client_id:
                continue
            key = (auth_type, mailbox, client_id)
            groups.setdefault(key, []).append(connector)

        connections_created = 0
        connectors_updated = 0
        connector_svc = ConnectorService(self._connectors)

        for (auth_type, mailbox, client_id), group in groups.items():
            provider = MailConnectionProvider(auth_type)
            merged_cfg: dict = {}
            for connector in group:
                merged_cfg.update(_connection_config_from_connector(connector.config, auth_type))
            label = f"{mailbox} ({provider.value.replace('_oauth', '')})"
            connection = self._conn_svc.upsert(
                tenant_id=tenant_id,
                connection_id=None,
                label=label,
                provider=provider.value,
                mailbox_email=mailbox,
                config_incoming=merged_cfg,
                active=True,
            )
            connections_created += 1
            for connector in group:
                cfg = strip_connector_oauth_fields(dict(connector.config))
                cfg["mail_connection_id"] = connection.id
                cfg["auth_type"] = connector_auth_type_for_connection(provider)
                connector_svc.upsert(
                    tenant_id=tenant_id,
                    direction=connector.direction,
                    type=connector.type,
                    mode=connector.mode,
                    config=cfg,
                    active=connector.active,
                )
                connectors_updated += 1

        self._session.flush()
        return MailConnectionMigrateResult(
            connections_created=connections_created,
            connectors_updated=connectors_updated,
            message=(
                f"Migrated {connectors_updated} connector(s) into {connections_created} mail connection(s)."
            ),
        )
