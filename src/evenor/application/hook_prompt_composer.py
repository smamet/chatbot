from __future__ import annotations

from evenor.automation.modules.registry import enabled_modules_for_tenant
from evenor.domain.constants import BASE_HOOK_FORMAT
from evenor.domain.models.tenant import Tenant, TenantConfig


def compose_hook_instructions(
    tenant: Tenant,
    *,
    active_integrations: set[str] | None = None,
) -> str:
    config = tenant.config
    module_ids = config.resolved_automation_modules(tenant.hook_instructions)
    integrations = active_integrations if active_integrations is not None else set()
    modules = enabled_modules_for_tenant(module_ids, active_integrations=integrations)
    parts: list[str] = [BASE_HOOK_FORMAT]
    for mod in modules:
        fragment = mod.prompt_fragment().strip()
        if fragment:
            parts.append(fragment)
    extra = config.hook_instructions_extra.strip()
    if not extra and tenant.hook_instructions and not config.automation_modules:
        extra = (tenant.hook_instructions or "").strip()
    if extra:
        parts.append(extra)
    return "\n\n".join(parts).strip()


def hooks_enabled_for_tenant(tenant: Tenant) -> bool:
    config = tenant.config
    if config.hook_instructions_extra.strip():
        return True
    if config.automation_modules:
        return True
    if config.resolved_automation_modules(tenant.hook_instructions):
        return True
    return bool((tenant.hook_instructions or "").strip())
