"""Helpers for trader bot access and settings resolution."""

from __future__ import annotations

from typing import Any

from chatbot.domain.models.tenant import Tenant, TraderSettings


def trader_settings_for(tenant: Tenant) -> TraderSettings:
    return tenant.config.trader


def trader_settings_as_integration_dict(tenant: Tenant) -> dict[str, Any]:
    """Shape expected by legacy CAC/live services that read integration config."""
    t = tenant.config.trader
    return {
        "symbol": t.symbol,
        "epic": t.epic,
        "fundmanager_url": t.fundmanager_url,
        "fundmanager_token": t.fundmanager_token,
        "max_open_positions": t.max_open_positions,
        "bot_id": tenant.slug,
        "market_profile": t.market_profile,
        "pnl_currency": t.pnl_currency,
        "point_value": t.point_value,
    }


def require_trader(tenant: Tenant) -> None:
    """Raise PermissionError if tenant is not a trader bot."""
    if not tenant.is_trader:
        raise PermissionError("Trading is only available for trader bots")
