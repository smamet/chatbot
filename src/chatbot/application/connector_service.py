from __future__ import annotations

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.domain.models.connector import (
    Connector,
    ConnectorDirection,
    ConnectorMode,
    ConnectorType,
)


class ConnectorService:
    def __init__(self, repository: SqlAlchemyConnectorRepository) -> None:
        self._repo = repository

    def list_for_tenant(self, tenant_id: int) -> list[Connector]:
        return self._repo.list_for_tenant(tenant_id)

    def _channel_config(
        self,
        tenant_id: int,
        *,
        type: ConnectorType,
        outbound: bool = False,
    ) -> dict | None:
        direction = ConnectorDirection.OUT if outbound else ConnectorDirection.IN
        c = self._repo.find_active(tenant_id, direction=direction, type=type)
        if c is None and not outbound:
            c = self._repo.find_active(tenant_id, direction=ConnectorDirection.OUT, type=type)
        return c.config if c else None

    def get_whatsapp_config(self, tenant_id: int, *, outbound: bool = False) -> dict | None:
        return self._channel_config(tenant_id, type=ConnectorType.WHATSAPP, outbound=outbound)

    def get_messenger_config(self, tenant_id: int, *, outbound: bool = False) -> dict | None:
        return self._channel_config(tenant_id, type=ConnectorType.MESSENGER, outbound=outbound)

    def get_instagram_config(self, tenant_id: int, *, outbound: bool = False) -> dict | None:
        return self._channel_config(tenant_id, type=ConnectorType.INSTAGRAM, outbound=outbound)

    def get_email_config(self, tenant_id: int, *, outbound: bool = False) -> dict | None:
        return self._channel_config(tenant_id, type=ConnectorType.EMAIL, outbound=outbound)

    def get(self, connector_id: int) -> Connector | None:
        return self._repo.find_by_id(connector_id)

    def find(
        self,
        tenant_id: int,
        *,
        direction: ConnectorDirection,
        type: ConnectorType,
    ) -> Connector | None:
        return self._repo.find_by_tenant_direction_type(tenant_id, direction=direction, type=type)

    def delete(self, connector_id: int) -> bool:
        return self._repo.delete(connector_id)

    def set_active(self, connector_id: int, active: bool) -> Connector | None:
        return self._repo.update(connector_id, active=active)

    def upsert(
        self,
        *,
        tenant_id: int,
        direction: ConnectorDirection,
        type: ConnectorType,
        mode: ConnectorMode,
        config: dict,
        active: bool = True,
    ) -> Connector:
        existing = self._repo.find_by_tenant_direction_type(
            tenant_id, direction=direction, type=type
        )
        if existing:
            updated = self._repo.update(existing.id, config=config, active=active, mode=mode)
            return updated or existing
        return self._repo.create(
            tenant_id=tenant_id,
            direction=direction,
            type=type,
            mode=mode,
            config=config,
            active=active,
        )
