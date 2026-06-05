from __future__ import annotations

from typing import Protocol

from chatbot.domain.models.tenant import Tenant, TenantConfig


class TenantRepository(Protocol):
    def find_by_id(self, tenant_id: int) -> Tenant | None: ...

    def find_by_slug(self, slug: str) -> Tenant | None: ...

    def find_by_token_hash(self, token_hash: str) -> Tenant | None: ...

    def list_all(self) -> list[Tenant]: ...

    def create(
        self,
        *,
        slug: str,
        name: str,
        token_hash: str,
        prompt: str,
        config: TenantConfig,
    ) -> Tenant: ...

    def update(
        self,
        tenant_id: int,
        *,
        name: str | None = None,
        prompt: str | None = None,
        config: TenantConfig | None = None,
        active: bool | None = None,
        token_hash: str | None = None,
    ) -> Tenant | None: ...

    def delete(self, tenant_id: int) -> bool: ...
