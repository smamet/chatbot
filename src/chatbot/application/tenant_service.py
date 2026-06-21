from __future__ import annotations

import hashlib
import re
import secrets
import shutil
import unicodedata
from dataclasses import replace
from decimal import Decimal

from chatbot.config.settings import Settings
from chatbot.domain.constants import DEFAULT_HOOK_INSTRUCTIONS
from chatbot.domain.contracts.tenant_repository import TenantRepository
from chatbot.domain.models.tenant import Tenant, TenantConfig, TenantCreateResult


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:64] or "tenant"


class TenantService:
    def __init__(self, repository: TenantRepository) -> None:
        self._repo = repository

    def list_tenants(self) -> list[Tenant]:
        return self._repo.list_all()

    def get_by_slug(self, slug: str) -> Tenant | None:
        return self._repo.find_by_slug(slug)

    def get_by_id(self, tenant_id: int) -> Tenant | None:
        return self._repo.find_by_id(tenant_id)

    def get_by_token(self, token: str) -> Tenant | None:
        return self._repo.find_by_token_hash(hash_token(token))

    def create_tenant(
        self,
        *,
        name: str,
        slug: str | None = None,
        prompt: str = "You are a helpful assistant.",
        config: TenantConfig | None = None,
        hook_instructions: str | None = None,
        gemini_api_key: str | None = None,
    ) -> TenantCreateResult:
        base_slug = slugify(slug or name)
        candidate = base_slug
        n = 0
        while self._repo.find_by_slug(candidate) is not None:
            n += 1
            candidate = f"{base_slug}-{n}"[:64]
        token = secrets.token_urlsafe(32)
        tenant = self._repo.create(
            slug=candidate,
            name=name.strip(),
            token_hash=hash_token(token),
            prompt=prompt.strip(),
            config=config or TenantConfig(),
            hook_instructions=hook_instructions,
            gemini_api_key=gemini_api_key,
        )
        return TenantCreateResult(tenant=tenant, token=token)

    def update_tenant(
        self,
        tenant_id: int,
        *,
        name: str | None = None,
        prompt: str | None = None,
        config: TenantConfig | None = None,
        active: bool | None = None,
        hook_instructions: str | None = None,
        update_hook_instructions: bool = False,
        gemini_api_key: str | None = None,
        update_gemini_api_key: bool = False,
    ) -> Tenant | None:
        return self._repo.update(
            tenant_id,
            name=name,
            prompt=prompt,
            config=config,
            active=active,
            hook_instructions=hook_instructions,
            update_hook_instructions=update_hook_instructions,
            gemini_api_key=gemini_api_key,
            update_gemini_api_key=update_gemini_api_key,
        )

    def regenerate_token(self, tenant_id: int) -> tuple[Tenant, str] | None:
        token = secrets.token_urlsafe(32)
        tenant = self._repo.update(tenant_id, token_hash=hash_token(token))
        if tenant is None:
            return None
        return tenant, token

    def update_client_billing(
        self,
        tenant_id: int,
        *,
        input_per_million_usd: Decimal | None,
        output_per_million_usd: Decimal | None,
    ) -> Tenant | None:
        return self._repo.update(
            tenant_id,
            client_billing_input_per_million_usd=input_per_million_usd,
            client_billing_output_per_million_usd=output_per_million_usd,
            update_client_billing=True,
        )

    def update_blocked_senders(self, tenant_id: int, blocked_senders: list[str]) -> Tenant | None:
        tenant = self._repo.find_by_id(tenant_id)
        if tenant is None:
            return None
        normalized = sorted(
            {
                addr.strip().lower()
                for addr in blocked_senders
                if addr and addr.strip()
            }
        )
        config = replace(tenant.config, email_blocked_senders=tuple(normalized))
        return self._repo.update(tenant_id, config=config)

    def add_blocked_sender(self, tenant_id: int, addr: str) -> Tenant | None:
        tenant = self._repo.find_by_id(tenant_id)
        if tenant is None:
            return None
        key = addr.strip().lower()
        if not key:
            return tenant
        merged = sorted(set(tenant.config.email_blocked_senders) | {key})
        config = replace(tenant.config, email_blocked_senders=tuple(merged))
        return self._repo.update(tenant_id, config=config)

    def unblock_sender(self, tenant_id: int, addr: str) -> Tenant | None:
        tenant = self._repo.find_by_id(tenant_id)
        if tenant is None:
            return None
        key = addr.strip().lower()
        remaining = tuple(a for a in tenant.config.email_blocked_senders if a != key)
        config = replace(tenant.config, email_blocked_senders=remaining)
        return self._repo.update(tenant_id, config=config)

    def delete_tenant(self, tenant_id: int, *, settings: Settings) -> bool:
        tenant = self._repo.find_by_id(tenant_id)
        if tenant is None:
            return False
        slug = tenant.slug
        if not self._repo.delete(tenant_id):
            return False
        for root in (settings.data_root / "docs" / slug, settings.lancedb_root / slug):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
        return True

    @staticmethod
    def default_hook_instructions() -> str:
        return DEFAULT_HOOK_INSTRUCTIONS
