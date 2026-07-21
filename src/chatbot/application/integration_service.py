from __future__ import annotations

from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.domain.models.integration import Integration, IntegrationType


class IntegrationService:
    def __init__(self, repository: SqlAlchemyIntegrationRepository) -> None:
        self._repo = repository

    def list_for_tenant(self, tenant_id: int) -> list[Integration]:
        return self._repo.list_for_tenant(tenant_id)

    def active_types_for_tenant(self, tenant_id: int) -> set[str]:
        return {i.type.value for i in self.list_for_tenant(tenant_id) if i.active}

    def get(self, integration_id: int) -> Integration | None:
        return self._repo.find_by_id(integration_id)

    def find(self, tenant_id: int, *, type: IntegrationType) -> Integration | None:
        return self._repo.find_by_tenant_type(tenant_id, type=type)

    def find_active(self, tenant_id: int, *, type: IntegrationType) -> Integration | None:
        return self._repo.find_active(tenant_id, type=type)

    def delete(self, integration_id: int) -> bool:
        return self._repo.delete(integration_id)

    def set_active(self, integration_id: int, active: bool) -> Integration | None:
        return self._repo.update(integration_id, active=active)

    def upsert(
        self,
        *,
        tenant_id: int,
        type: IntegrationType,
        config: dict,
        active: bool = True,
    ) -> Integration:
        existing = self._repo.find_by_tenant_type(tenant_id, type=type)
        if existing:
            updated = self._repo.update(existing.id, config=config, active=active)
            return updated or existing
        return self._repo.create(
            tenant_id=tenant_id,
            type=type,
            config=config,
            active=active,
        )
