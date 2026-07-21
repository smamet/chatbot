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

    def find_ig(self, tenant_id: int, *, active_only: bool = False) -> Connector | None:
        """Prefer bidirectional IG; fall back to legacy out then in."""
        for direction in (
            ConnectorDirection.BOTH,
            ConnectorDirection.OUT,
            ConnectorDirection.IN,
        ):
            if active_only:
                found = self._repo.find_active(
                    tenant_id, direction=direction, type=ConnectorType.IG
                )
            else:
                found = self._repo.find_by_tenant_direction_type(
                    tenant_id, direction=direction, type=ConnectorType.IG
                )
            if found is not None:
                return found
        return None

    def get_ig_config(self, tenant_id: int) -> dict | None:
        connector = self.find_ig(tenant_id, active_only=True)
        return connector.config if connector else None

    def migrate_ig_to_both(self, tenant_id: int) -> bool:
        """
        Collapse legacy ig:in / ig:out into a single ig:both row.
        Returns True when rows were created or deleted.
        """
        both = self._repo.find_by_tenant_direction_type(
            tenant_id, direction=ConnectorDirection.BOTH, type=ConnectorType.IG
        )
        legacy: list[Connector] = []
        for direction in (ConnectorDirection.IN, ConnectorDirection.OUT):
            found = self._repo.find_by_tenant_direction_type(
                tenant_id, direction=direction, type=ConnectorType.IG
            )
            if found is not None:
                legacy.append(found)
        if not legacy:
            return False

        source = max(legacy, key=lambda c: c.updated_at)
        active = both.active if both is not None else any(c.active for c in legacy)
        if both is None:
            self._repo.create(
                tenant_id=tenant_id,
                direction=ConnectorDirection.BOTH,
                type=ConnectorType.IG,
                mode=ConnectorMode.DIRECT,
                config=dict(source.config),
                active=active,
            )
        for connector in legacy:
            self._repo.delete(connector.id)
        return True

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

    def upsert_ig(
        self,
        *,
        tenant_id: int,
        config: dict,
        active: bool = True,
    ) -> Connector:
        """Save IG as a single bidirectional connector and retire legacy in/out rows."""
        saved = self.upsert(
            tenant_id=tenant_id,
            direction=ConnectorDirection.BOTH,
            type=ConnectorType.IG,
            mode=ConnectorMode.DIRECT,
            config=config,
            active=active,
        )
        self.migrate_ig_to_both(tenant_id)
        return (
            self.find(tenant_id, direction=ConnectorDirection.BOTH, type=ConnectorType.IG)
            or saved
        )
