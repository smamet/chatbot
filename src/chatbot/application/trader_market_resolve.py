"""Resolve bot symbol → epic + instrument economics on create/update."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.domain.models.tenant import TraderSettings
from chatbot.trader.epic_resolve import alias_epic_for_symbol, resolve_ticker_to_epic
from chatbot.trader.ig_connector import IgConnector
from chatbot.trader.ig_ohlc import ig_config_from_connector
from chatbot.trader.instrument_economics import (
    resolve_economics_from_ig,
    resolve_instrument_economics,
)
from chatbot.trader.profiles import get_profile

logger = logging.getLogger(__name__)


def _primary_ig_config(session: Session | None, tenant_id: int | None) -> dict[str, Any] | None:
    if session is None or tenant_id is None:
        return None
    try:
        svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        return svc.get_ig_config(tenant_id)
    except Exception:
        logger.debug("No IG connector for tenant %s", tenant_id, exc_info=True)
        return None


def resolve_trader_market_settings(
    *,
    symbol: str,
    market_profile: str = "cac40",
    explicit_epic: str | None = None,
    session: Session | None = None,
    tenant_id: int | None = None,
    legacy_connector_epic: str | None = None,
) -> dict[str, Any]:
    """
    Resolve symbol → epic + point_value + pnl_currency.

    Returns keys: symbol, epic, market_profile, pnl_currency, point_value,
    resolved_name, resolve_source.
    """
    profile = get_profile(market_profile)
    sym = (symbol or profile.default_symbol).strip() or profile.default_symbol
    # Prefer explicit full epic; else legacy connector epic during migration.
    seed_epic = (explicit_epic or "").strip() or (legacy_connector_epic or "").strip()

    ig_cfg = _primary_ig_config(session, tenant_id)
    ig: IgConnector | None = None
    if ig_cfg and ig_cfg.get("api_key") and ig_cfg.get("username") and ig_cfg.get("password"):
        try:
            # Epic placeholder until resolve finishes.
            cfg = ig_config_from_connector(ig_cfg, epic=seed_epic or profile.default_epic)
            ig = IgConnector(cfg, dry_run=True)
            ig.login()
            if not ig.authenticated:
                ig = None
        except Exception:
            logger.info("IG login failed during market resolve; using offline fallback")
            ig = None

    resolved = resolve_ticker_to_epic(
        ig,
        sym,
        explicit_epic=seed_epic or None,
        profile_id=profile.id,
    )
    if resolved is None:
        epic = alias_epic_for_symbol(sym, profile_id=profile.id) or profile.default_epic
        source = "alias"
        name = epic
    else:
        epic = resolved.epic
        source = resolved.source
        name = resolved.name or resolved.epic

    if ig is not None:
        try:
            ig.config.epic = epic
            econ = resolve_economics_from_ig(
                ig, epic=epic, symbol=sym, profile_id=profile.id
            )
        except Exception:
            logger.debug("Economics resolve failed for %s", epic, exc_info=True)
            econ = resolve_instrument_economics(
                None, symbol=sym, profile_id=profile.id
            )
    else:
        econ = resolve_instrument_economics(None, symbol=sym, profile_id=profile.id)

    return {
        "symbol": sym,
        "epic": epic,
        "market_profile": profile.id,
        "pnl_currency": econ.currency,
        "point_value": float(econ.point_value),
        "resolved_name": name,
        "resolve_source": source,
        "economics_source": econ.source,
    }


def apply_resolved_to_trader_settings(
    current: TraderSettings,
    resolved: dict[str, Any],
    *,
    max_open_positions: int | None = None,
    fundmanager_url: str | None = None,
    fundmanager_token: str | None = None,
) -> TraderSettings:
    return TraderSettings(
        market_profile=str(resolved.get("market_profile") or current.market_profile),
        symbol=str(resolved.get("symbol") or current.symbol),
        epic=str(resolved.get("epic") or current.epic),
        fundmanager_url=(
            fundmanager_url
            if fundmanager_url is not None
            else current.fundmanager_url
        ),
        fundmanager_token=(
            fundmanager_token
            if fundmanager_token is not None
            else current.fundmanager_token
        ),
        max_open_positions=(
            max(1, int(max_open_positions))
            if max_open_positions is not None
            else current.max_open_positions
        ),
        pnl_currency=str(resolved.get("pnl_currency") or current.pnl_currency or ""),
        point_value=float(resolved.get("point_value") or current.point_value or 0.0),
    )
