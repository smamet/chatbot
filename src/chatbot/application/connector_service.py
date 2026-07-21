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

    def list_ig(self, tenant_id: int, *, active_only: bool = False) -> list[Connector]:
        """All IG connectors for a tenant (bidirectional first, then legacy in/out)."""
        out: list[Connector] = []
        seen: set[int] = set()
        for direction in (
            ConnectorDirection.BOTH,
            ConnectorDirection.OUT,
            ConnectorDirection.IN,
        ):
            rows = self._repo.list_by_tenant_direction_type(
                tenant_id, direction=direction, type=ConnectorType.IG
            )
            for row in rows:
                if row.id in seen:
                    continue
                if active_only and not row.active:
                    continue
                seen.add(row.id)
                out.append(row)
        return out

    def find_ig(self, tenant_id: int, *, active_only: bool = False) -> Connector | None:
        """Prefer first bidirectional IG; fall back to legacy out then in."""
        rows = self.list_ig(tenant_id, active_only=active_only)
        return rows[0] if rows else None

    def get_ig_config(self, tenant_id: int) -> dict | None:
        connector = self.find_ig(tenant_id, active_only=True)
        return connector.config if connector else None

    def get_ig_by_id(
        self, tenant_id: int, connector_id: int, *, active_only: bool = False
    ) -> Connector | None:
        connector = self._repo.find_by_id(connector_id)
        if connector is None or connector.tenant_id != tenant_id:
            return None
        if connector.type != ConnectorType.IG:
            return None
        if active_only and not connector.active:
            return None
        return connector

    def migrate_ig_to_both(self, tenant_id: int) -> bool:
        """
        Collapse legacy ig:in / ig:out into a bidirectional row.
        Never deletes existing ig:both rows (multi-account support).
        """
        both_rows = self._repo.list_by_tenant_direction_type(
            tenant_id, direction=ConnectorDirection.BOTH, type=ConnectorType.IG
        )
        legacy: list[Connector] = []
        for direction in (ConnectorDirection.IN, ConnectorDirection.OUT):
            legacy.extend(
                self._repo.list_by_tenant_direction_type(
                    tenant_id, direction=direction, type=ConnectorType.IG
                )
            )
        if not legacy:
            return False

        source = max(legacy, key=lambda c: c.updated_at)
        active = any(c.active for c in both_rows) or any(c.active for c in legacy)
        if not both_rows:
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

    def create_ig(
        self,
        *,
        tenant_id: int,
        config: dict,
        active: bool = True,
    ) -> Connector:
        """Always insert a new bidirectional IG connector (multi-account)."""
        self.migrate_ig_to_both(tenant_id)
        return self._repo.create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.BOTH,
            type=ConnectorType.IG,
            mode=ConnectorMode.DIRECT,
            config=config,
            active=active,
        )

    def update_ig(
        self,
        connector_id: int,
        *,
        tenant_id: int,
        config: dict,
        active: bool = True,
    ) -> Connector | None:
        existing = self.get_ig_by_id(tenant_id, connector_id)
        if existing is None:
            return None
        return self._repo.update(
            connector_id,
            config=config,
            active=active,
            mode=ConnectorMode.DIRECT,
        )

    def upsert_ig(
        self,
        *,
        tenant_id: int,
        config: dict,
        active: bool = True,
        connector_id: int | None = None,
    ) -> Connector:
        """
        Update an IG connector by id, or update the first / create one.
        Never deletes extra ig:both rows.
        """
        self.migrate_ig_to_both(tenant_id)
        if connector_id is not None:
            updated = self.update_ig(
                connector_id, tenant_id=tenant_id, config=config, active=active
            )
            if updated is not None:
                return updated
            return self.create_ig(tenant_id=tenant_id, config=config, active=active)
        existing = self.find(tenant_id, direction=ConnectorDirection.BOTH, type=ConnectorType.IG)
        if existing is not None:
            return (
                self._repo.update(
                    existing.id,
                    config=config,
                    active=active,
                    mode=ConnectorMode.DIRECT,
                )
                or existing
            )
        return self.create_ig(tenant_id=tenant_id, config=config, active=active)
