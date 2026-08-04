from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.mail_connection_repository import SqlAlchemyMailConnectionRepository
from evenor.application.mail_oauth_service import (
    apply_oauth_tokens_to_config,
    has_oauth_refresh_token,
    prepare_oauth_mail_config,
    resolve_mail_oauth_credentials,
)
from evenor.config.settings import Settings, get_settings
from evenor.domain.models.connector import Connector, ConnectorType
from evenor.domain.models.connector_schema import oauth_managed_connector_keys, runtime_mail_config_keys, secret_connector_keys
from evenor.domain.models.mail_connection import MailConnection, MailConnectionProvider
from evenor.domain.models.mail_connection_presets import build_runtime_mail_config, provider_auth_type


class MailConnectionError(RuntimeError):
    pass


_CONNECTION_SECRET_KEYS = frozenset(
    {
        "microsoft_client_secret",
        "google_client_secret",
        "oauth_refresh_token",
        "oauth_access_token",
    }
)


@dataclass(frozen=True, slots=True)
class MailConnectionClientView:
    id: int
    label: str
    provider: str
    mailbox_email: str
    active: bool
    oauth_connected: bool
    config: dict[str, Any]


def _merge_connection_config(existing: dict | None, incoming: dict) -> dict:
    preserve_if_empty = _CONNECTION_SECRET_KEYS | oauth_managed_connector_keys()
    base = dict(existing or {})
    for key, value in incoming.items():
        if key in preserve_if_empty and not str(value).strip():
            continue
        base[key] = value
    return base


def _client_config(connection: MailConnection) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, value in connection.config.items():
        if key in _CONNECTION_SECRET_KEYS:
            cfg[key] = ""
        else:
            cfg[key] = value
    return cfg


class MailConnectionService:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._repo = SqlAlchemyMailConnectionRepository(session)
        self._connectors = SqlAlchemyConnectorRepository(session)

    def list_for_tenant(self, tenant_id: int) -> list[MailConnection]:
        return self._repo.list_for_tenant(tenant_id)

    def list_client_views(self, tenant_id: int) -> list[MailConnectionClientView]:
        return [self._to_client_view(c) for c in self.list_for_tenant(tenant_id)]

    def get_for_tenant(self, connection_id: int, tenant_id: int) -> MailConnection | None:
        return self._repo.find_for_tenant(connection_id, tenant_id)

    def upsert(
        self,
        *,
        tenant_id: int,
        connection_id: int | None,
        label: str,
        provider: str,
        mailbox_email: str,
        config_incoming: dict,
        active: bool = True,
    ) -> MailConnection:
        try:
            provider_enum = MailConnectionProvider(provider)
        except ValueError as exc:
            raise MailConnectionError("Invalid mail connection provider") from exc
        label = label.strip()
        mailbox_email = mailbox_email.strip().lower()
        if not label:
            raise MailConnectionError("Label is required")
        if not mailbox_email:
            raise MailConnectionError("Mailbox email is required")
        existing = self._repo.find_for_tenant(connection_id, tenant_id) if connection_id else None
        merged = _merge_connection_config(existing.config if existing else None, config_incoming)
        if existing is None:
            return self._repo.create(
                tenant_id=tenant_id,
                label=label,
                provider=provider_enum,
                mailbox_email=mailbox_email,
                config=merged,
                active=active,
            )
        updated = self._repo.update(
            existing.id,
            label=label,
            mailbox_email=mailbox_email,
            config=merged,
            active=active,
        )
        if updated is None:
            raise MailConnectionError("Mail connection not found")
        return updated

    def apply_oauth_tokens(self, connection: MailConnection, tokens: Any) -> MailConnection:
        cfg = apply_oauth_tokens_to_config(connection.config, tokens)
        updated = self._repo.update(connection.id, config=cfg)
        if updated is None:
            raise MailConnectionError("Mail connection not found")
        return updated

    def update_config(self, connection_id: int, config: dict) -> MailConnection | None:
        return self._repo.update(connection_id, config=config)

    def referencing_connectors(self, tenant_id: int, connection_id: int) -> list[Connector]:
        refs: list[Connector] = []
        for connector in self._connectors.list_for_tenant(tenant_id):
            if connector.type != ConnectorType.EMAIL:
                continue
            raw = connector.config.get("mail_connection_id")
            try:
                ref_id = int(raw)
            except (TypeError, ValueError):
                continue
            if ref_id == connection_id:
                refs.append(connector)
        return refs

    def delete(self, tenant_id: int, connection_id: int) -> None:
        connection = self._repo.find_for_tenant(connection_id, tenant_id)
        if connection is None:
            raise MailConnectionError("Mail connection not found")
        refs = [c for c in self.referencing_connectors(tenant_id, connection_id) if c.active]
        if refs:
            raise MailConnectionError(
                "Cannot delete: connection is used by active email connector(s). "
                "Remove the reference first."
            )
        if not self._repo.delete(connection_id):
            raise MailConnectionError("Mail connection not found")

    def resolve_runtime_config(
        self,
        connection: MailConnection,
        *,
        direction: str,
        refresh: bool = True,
        force_oauth_refresh: bool = False,
        settings: Settings | None = None,
    ) -> tuple[dict, dict | None]:
        resolved_settings = settings or get_settings()
        runtime = build_runtime_mail_config(connection, direction=direction)
        if not refresh:
            return runtime, None
        mail_cfg, updated = prepare_oauth_mail_config(
            runtime,
            direction=direction,
            settings=resolved_settings,
            force_refresh=force_oauth_refresh,
        )
        if updated is None:
            return mail_cfg, None
        cfg = dict(connection.config)
        for key in oauth_managed_connector_keys():
            if key in updated:
                cfg[key] = updated[key]
        persisted = self._repo.update(connection.id, config=cfg)
        if persisted is None:
            return mail_cfg, None
        return mail_cfg, persisted.config

    def resolve_for_connector(
        self,
        connector: Connector,
        *,
        direction: str,
        settings: Settings | None = None,
        force_oauth_refresh: bool = False,
    ) -> dict:
        raw_id = connector.config.get("mail_connection_id")
        try:
            connection_id = int(raw_id)
        except (TypeError, ValueError):
            raise MailConnectionError("Email connector is missing mail_connection_id")
        connection = self._repo.find_for_tenant(connection_id, connector.tenant_id)
        if connection is None:
            raise MailConnectionError("Mail connection not found")
        if not connection.active:
            raise MailConnectionError("Mail connection is inactive")
        mail_cfg, _updated = self.resolve_runtime_config(
            connection,
            direction=direction,
            settings=settings,
            force_oauth_refresh=force_oauth_refresh,
        )
        return {**mail_cfg, **connector_mail_overlay(connector)}

    def _to_client_view(self, connection: MailConnection) -> MailConnectionClientView:
        return MailConnectionClientView(
            id=connection.id,
            label=connection.label,
            provider=connection.provider.value,
            mailbox_email=connection.mailbox_email,
            active=connection.active,
            oauth_connected=has_oauth_refresh_token(connection.config),
            config=_client_config(connection),
        )


def _oauth_subset_from_runtime(runtime: dict) -> dict:
    keys = oauth_managed_connector_keys()
    return {key: runtime[key] for key in keys if key in runtime}


def connection_client_credentials(
    connection: MailConnection,
    settings: Settings,
) -> tuple[str, str]:
    return resolve_mail_oauth_credentials(connection, settings)


_CONNECTOR_SEND_OVERLAY_KEYS = frozenset({"from_addr", "default_subject", "outbound_provider"})


def connector_mail_overlay(connector: Connector) -> dict[str, Any]:
    """Connector-owned send settings that are not stored on the mail connection."""
    overlay: dict[str, Any] = {}
    for key in _CONNECTOR_SEND_OVERLAY_KEYS:
        value = connector.config.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        overlay[key] = value
    return overlay


def strip_connector_oauth_fields(config: dict) -> dict:
    """Remove inline OAuth/IMAP/SMTP host fields when using a mail connection."""
    strip_keys = (
        secret_connector_keys()
        | oauth_managed_connector_keys()
        | runtime_mail_config_keys()
        | frozenset(
            {
                "imap_host",
                "imap_port",
                "imap_use_ssl",
                "username",
                "password",
                "smtp_host",
                "smtp_port",
                "smtp_username",
                "smtp_password",
                "smtp_use_tls",
                "microsoft_client_id",
                "microsoft_client_secret",
                "google_client_id",
                "google_client_secret",
            }
        )
    )
    return {key: value for key, value in config.items() if key not in strip_keys}


def connector_auth_type_for_connection(provider: MailConnectionProvider) -> str:
    return provider_auth_type(provider)
