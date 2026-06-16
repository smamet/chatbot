from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from chatbot.application.gemini_pricing import (
    ModelPricing,
    client_billing_rates,
    internal_pricing_table,
    row_cost_usd,
)
from chatbot.config.settings import Settings
from chatbot.domain.models.api_usage import ApiUsageDayEntry
from chatbot.domain.models.tenant import Tenant

CostProfile = Literal["internal", "client"]


@dataclass(frozen=True, slots=True)
class CostDayPoint:
    usage_date: date
    cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class CostEstimate:
    total_usd: Decimal
    daily: tuple[CostDayPoint, ...]
    unknown_models: frozenset[str]


@dataclass(frozen=True, slots=True)
class UsageDailyRowCost:
    entry: ApiUsageDayEntry
    cost_usd: Decimal
    unknown_model: bool


def _pricing_for_profile(
    profile: CostProfile,
    settings: Settings,
    tenant: Tenant | None,
) -> tuple[dict[str, ModelPricing] | None, ModelPricing, bool]:
    if profile == "internal":
        table = internal_pricing_table(settings)
        fallback = table.get("gemini-2.5-flash", next(iter(table.values())))
        return table, fallback, True
    if tenant is None:
        raise ValueError("tenant required for client cost profile")
    flat = client_billing_rates(settings, tenant)
    return None, flat, False


class UsageCostService:
    def estimate_cost(
        self,
        entries: list[ApiUsageDayEntry],
        *,
        profile: CostProfile,
        settings: Settings,
        tenant: Tenant | None = None,
    ) -> CostEstimate:
        internal_table, flat, use_model_rates = _pricing_for_profile(profile, settings, tenant)
        unknown: set[str] = set()
        total = Decimal("0")
        by_day: dict[date, Decimal] = {}

        for entry in entries:
            cost, is_unknown = row_cost_usd(
                prompt_tokens=entry.prompt_tokens,
                output_tokens=entry.output_tokens,
                model=entry.model,
                pricing=flat,
                internal_table=internal_table,
                use_model_rates=use_model_rates,
            )
            if is_unknown:
                unknown.add(entry.model)
            total += cost
            by_day[entry.usage_date] = by_day.get(entry.usage_date, Decimal("0")) + cost

        daily = tuple(
            CostDayPoint(usage_date=d, cost_usd=by_day[d])
            for d in sorted(by_day)
        )
        return CostEstimate(total_usd=total, daily=daily, unknown_models=frozenset(unknown))

    def row_costs(
        self,
        entries: list[ApiUsageDayEntry],
        *,
        profile: CostProfile,
        settings: Settings,
        tenant: Tenant | None = None,
    ) -> list[UsageDailyRowCost]:
        internal_table, flat, use_model_rates = _pricing_for_profile(profile, settings, tenant)
        rows: list[UsageDailyRowCost] = []
        for entry in entries:
            cost, is_unknown = row_cost_usd(
                prompt_tokens=entry.prompt_tokens,
                output_tokens=entry.output_tokens,
                model=entry.model,
                pricing=flat,
                internal_table=internal_table,
                use_model_rates=use_model_rates,
            )
            rows.append(UsageDailyRowCost(entry=entry, cost_usd=cost, unknown_model=is_unknown))
        return rows
