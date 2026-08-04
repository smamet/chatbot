from __future__ import annotations

from datetime import date, datetime, UTC
from decimal import Decimal

import pytest

from evenor.application.usage_cost_service import UsageCostService
from evenor.config.settings import Settings
from evenor.domain.models.api_usage import ApiUsageDayEntry
from evenor.domain.models.tenant import Tenant, TenantConfig


def _settings() -> Settings:
    return Settings(
        gemini_api_key="k",
        admin_token="a",
        app_secret_key="s",
        session_secret="s",
        client_billing_input_per_million_usd=2.0,
        client_billing_output_per_million_usd=6.0,
    )


def _tenant(**kwargs) -> Tenant:
    now = datetime.now(UTC)
    base = dict(
        id=1,
        slug="demo",
        name="Demo",
        prompt="",
        hook_instructions=None,
        gemini_api_key=None,
        config=TenantConfig(),
        active=True,
        created_at=now,
        updated_at=now,
    )
    base.update(kwargs)
    return Tenant(**base)


def test_internal_cost_uses_model_rates() -> None:
    entries = [
        ApiUsageDayEntry(
            usage_date=date(2026, 6, 1),
            operation="chat",
            model="gemini-2.5-flash",
            prompt_tokens=1_000_000,
            output_tokens=0,
            total_tokens=1_000_000,
            call_count=1,
        )
    ]
    est = UsageCostService().estimate_cost(
        entries,
        profile="internal",
        settings=_settings(),
    )
    assert est.total_usd == Decimal("0.30")


def test_client_cost_uses_flat_tenant_rates() -> None:
    entries = [
        ApiUsageDayEntry(
            usage_date=date(2026, 6, 1),
            operation="chat",
            model="gemini-2.5-flash",
            prompt_tokens=1_000_000,
            output_tokens=1_000_000,
            total_tokens=2_000_000,
            call_count=2,
        )
    ]
    tenant = _tenant(
        client_billing_input_per_million_usd=Decimal("5"),
        client_billing_output_per_million_usd=Decimal("10"),
    )
    est = UsageCostService().estimate_cost(
        entries,
        profile="client",
        settings=_settings(),
        tenant=tenant,
    )
    assert est.total_usd == Decimal("15")
