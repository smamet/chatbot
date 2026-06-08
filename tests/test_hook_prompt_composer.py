from __future__ import annotations

from chatbot.application.hook_prompt_composer import compose_hook_instructions, hooks_enabled_for_tenant
from chatbot.automation.modules.registry import enabled_modules_for_tenant
from chatbot.domain.models.tenant import Tenant, TenantConfig


def test_compose_hook_includes_orders_module() -> None:
    from datetime import UTC, datetime

    tenant = Tenant(
        id=1,
        slug="t",
        name="T",
        prompt="Hi",
        hook_instructions=None,
        gemini_api_key=None,
        config=TenantConfig(automation_modules=("core.orders",)),
        active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    text = compose_hook_instructions(tenant)
    assert "order.create" in text
    assert "===HOOK===" in text


def test_erpnext_quote_requires_integration() -> None:
    mods = enabled_modules_for_tenant(
        ["core.orders", "erpnext.quote"],
        active_integrations=set(),
    )
    assert [m.id for m in mods] == ["core.orders"]


def test_hooks_enabled_with_modules() -> None:
    from datetime import UTC, datetime

    tenant = Tenant(
        id=1,
        slug="t",
        name="T",
        prompt="Hi",
        hook_instructions=None,
        gemini_api_key=None,
        config=TenantConfig(automation_modules=("core.orders",)),
        active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    assert hooks_enabled_for_tenant(tenant) is True
