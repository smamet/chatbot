from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from evenor.config.settings import Settings

# USD per 1M tokens — https://ai.google.dev/gemini-api/docs/pricing
DEFAULT_GEMINI_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "gemini-2.5-flash": (Decimal("0.30"), Decimal("2.50")),  # text/image/video input
    "gemini-2.0-flash": (Decimal("0.10"), Decimal("0.40")),
    "gemini-embedding-001": (Decimal("0.15"), Decimal("0")),
    "text-embedding-004": (Decimal("0.10"), Decimal("0")),
}


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal


def _parse_pricing_json(raw: str) -> dict[str, ModelPricing]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("GEMINI_PRICING_JSON must be a JSON object")
    out: dict[str, ModelPricing] = {}
    for model, rates in data.items():
        if not isinstance(rates, dict):
            continue
        out[str(model)] = ModelPricing(
            input_per_million_usd=Decimal(str(rates.get("input", 0))),
            output_per_million_usd=Decimal(str(rates.get("output", 0))),
        )
    return out


def internal_pricing_table(settings: Settings) -> dict[str, ModelPricing]:
    table = {
        model: ModelPricing(input_per_million_usd=inp, output_per_million_usd=out)
        for model, (inp, out) in DEFAULT_GEMINI_PRICING.items()
    }
    raw = (settings.gemini_pricing_json or "").strip()
    if raw:
        table.update(_parse_pricing_json(raw))
    return table


def client_billing_rates(settings: Settings, tenant) -> ModelPricing:
    inp = tenant.client_billing_input_per_million_usd
    out = tenant.client_billing_output_per_million_usd
    if inp is None:
        inp = Decimal(str(settings.client_billing_input_per_million_usd))
    if out is None:
        out = Decimal(str(settings.client_billing_output_per_million_usd))
    return ModelPricing(input_per_million_usd=inp, output_per_million_usd=out)


def row_cost_usd(
    *,
    prompt_tokens: int,
    output_tokens: int,
    model: str,
    pricing: ModelPricing,
    internal_table: dict[str, ModelPricing] | None = None,
    use_model_rates: bool,
) -> tuple[Decimal, bool]:
    unknown = False
    if use_model_rates and internal_table is not None:
        rates = internal_table.get(model)
        if rates is None:
            rates = internal_table.get("gemini-2.5-flash", pricing)
            unknown = model not in internal_table
    else:
        rates = pricing
    cost = (
        Decimal(prompt_tokens) / Decimal(1_000_000) * rates.input_per_million_usd
        + Decimal(output_tokens) / Decimal(1_000_000) * rates.output_per_million_usd
    )
    return cost, unknown
