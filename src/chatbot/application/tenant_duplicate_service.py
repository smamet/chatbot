"""Clone a bot's credentials and settings into a new tenant (no RAG / ops data)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.orm import UserBotAccessRow
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.application.integration_service import IntegrationService
from chatbot.application.mail_connection_service import MailConnectionService
from chatbot.application.tenant_service import TenantService
from chatbot.application.trader_live_service import (
    live_config_path,
    load_live_config,
    save_live_config,
)
from chatbot.application.user_service import UserService
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.tenant import BotType, TenantConfig, TenantCreateResult
from chatbot.trader.profiles import default_prompt_text


class TenantDuplicateError(ValueError):
    pass


def _copy_config(config: TenantConfig) -> TenantConfig:
    return TenantConfig.from_json(config.to_json())


def _apply_trader_overrides(
    config: TenantConfig,
    *,
    market_profile: str | None,
    symbol: str | None,
    epic: str | None,
) -> TenantConfig:
    trader = config.trader
    updates: dict[str, Any] = {}
    if market_profile is not None and str(market_profile).strip():
        updates["market_profile"] = str(market_profile).strip()
    if symbol is not None and str(symbol).strip():
        updates["symbol"] = str(symbol).strip()
    if epic is not None and str(epic).strip():
        updates["epic"] = str(epic).strip()
    if not updates:
        return config
    return replace(config, trader=replace(trader, **updates))


def _clone_mail_connections(
    mail_svc: MailConnectionService,
    *,
    source_tenant_id: int,
    dest_tenant_id: int,
) -> dict[int, int]:
    """Copy mail connections; return old_id → new_id."""
    id_map: dict[int, int] = {}
    for src in mail_svc.list_for_tenant(source_tenant_id):
        created = mail_svc.upsert(
            tenant_id=dest_tenant_id,
            connection_id=None,
            label=src.label,
            provider=src.provider.value if hasattr(src.provider, "value") else str(src.provider),
            mailbox_email=src.mailbox_email,
            config_incoming=dict(src.config or {}),
            active=bool(src.active),
        )
        id_map[src.id] = created.id
    return id_map


def _remap_mail_connection_id(config: dict[str, Any], mail_id_map: dict[int, int]) -> dict[str, Any]:
    out = dict(config)
    raw = out.get("mail_connection_id")
    if raw is None or raw == "":
        return out
    try:
        old_id = int(raw)
    except (TypeError, ValueError):
        return out
    if old_id in mail_id_map:
        out["mail_connection_id"] = mail_id_map[old_id]
    return out


def _clone_connectors(
    connector_svc: ConnectorService,
    *,
    source_tenant_id: int,
    dest_tenant_id: int,
    mail_id_map: dict[int, int],
) -> dict[int, int]:
    """Copy connectors (incl. secrets); return old_id → new_id."""
    id_map: dict[int, int] = {}
    for src in connector_svc.list_for_tenant(source_tenant_id):
        cfg = _remap_mail_connection_id(dict(src.config or {}), mail_id_map)
        if src.type == ConnectorType.IG and src.direction == ConnectorDirection.BOTH:
            created = connector_svc.create_ig(
                tenant_id=dest_tenant_id,
                config=cfg,
                active=bool(src.active),
            )
        else:
            created = connector_svc.upsert(
                tenant_id=dest_tenant_id,
                direction=src.direction,
                type=src.type,
                mode=src.mode if isinstance(src.mode, ConnectorMode) else ConnectorMode.DIRECT,
                config=cfg,
                active=bool(src.active),
            )
        id_map[src.id] = created.id
    return id_map


def _clone_integrations(
    integration_svc: IntegrationService,
    *,
    source_tenant_id: int,
    dest_tenant_id: int,
) -> None:
    for src in integration_svc.list_for_tenant(source_tenant_id):
        integration_svc.upsert(
            tenant_id=dest_tenant_id,
            type=src.type,
            config=dict(src.config or {}),
            active=bool(src.active),
        )


def _clone_user_access(session: Session, *, source_tenant_id: int, dest_tenant_id: int) -> None:
    user_ids = list(
        session.scalars(
            select(UserBotAccessRow.user_id).where(UserBotAccessRow.tenant_id == source_tenant_id)
        )
    )
    if not user_ids:
        return
    user_svc = UserService(SqlAlchemyUserRepository(session))
    for user_id in user_ids:
        user_svc.grant_access(user_id, dest_tenant_id)


def _clone_live_config(
    settings: Settings,
    *,
    source_slug: str,
    dest_slug: str,
    connector_id_map: dict[int, int],
) -> None:
    path = live_config_path(settings, source_slug)
    if not path.is_file():
        return
    live = load_live_config(settings, source_slug)
    remapped: list[int] = []
    for raw in live.get("ig_connector_ids") or []:
        try:
            old_id = int(raw)
        except (TypeError, ValueError):
            continue
        new_id = connector_id_map.get(old_id)
        if new_id is not None:
            remapped.append(new_id)
    strategy = live.get("strategy") if isinstance(live.get("strategy"), dict) else {}
    save_live_config(
        settings,
        dest_slug,
        {
            "mode": "off",
            "ig_connector_ids": remapped,
            "strategy": dict(strategy),
        },
    )


def duplicate_tenant(
    session: Session,
    settings: Settings,
    source_slug: str,
    *,
    name: str,
    slug: str | None = None,
    market_profile: str | None = None,
    symbol: str | None = None,
    epic: str | None = None,
    reset_prompt_from_profile: bool = False,
) -> TenantCreateResult:
    """
    Create a new bot with the source's credentials and settings.

    Does not copy docs, LanceDB, catalog, operational tables, or trader OHLC/journals.
    """
    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    source = tenant_svc.get_by_slug(source_slug)
    if source is None:
        raise TenantDuplicateError(f"Bot not found: {source_slug}")

    create_name = (name or "").strip() or f"{source.name} (copy)"
    config = _copy_config(source.config)
    if source.bot_type == BotType.TRADER:
        config = _apply_trader_overrides(
            config,
            market_profile=market_profile,
            symbol=symbol,
            epic=epic,
        )

    prompt = source.prompt
    if reset_prompt_from_profile and source.bot_type == BotType.TRADER:
        profile_id = config.trader.market_profile
        profile_prompt = default_prompt_text(profile_id)
        if profile_prompt.strip():
            prompt = profile_prompt

    result = tenant_svc.create_tenant(
        name=create_name,
        slug=(slug.strip() if slug else None) or None,
        prompt=prompt,
        config=config,
        hook_instructions=source.hook_instructions,
        gemini_api_key=source.gemini_api_key,
        bot_type=source.bot_type,
    )
    dest = result.tenant

    # Preserve active flag from source (create_tenant leaves default active).
    if source.active != dest.active:
        updated = tenant_svc.update_tenant(dest.id, active=source.active)
        if updated is not None:
            dest = updated
            result = TenantCreateResult(tenant=dest, token=result.token)

    mail_svc = MailConnectionService(session)
    mail_id_map = _clone_mail_connections(
        mail_svc,
        source_tenant_id=source.id,
        dest_tenant_id=dest.id,
    )

    connector_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    connector_id_map = _clone_connectors(
        connector_svc,
        source_tenant_id=source.id,
        dest_tenant_id=dest.id,
        mail_id_map=mail_id_map,
    )

    _clone_integrations(
        IntegrationService(SqlAlchemyIntegrationRepository(session)),
        source_tenant_id=source.id,
        dest_tenant_id=dest.id,
    )
    _clone_user_access(session, source_tenant_id=source.id, dest_tenant_id=dest.id)

    if source.bot_type == BotType.TRADER:
        _clone_live_config(
            settings,
            source_slug=source.slug,
            dest_slug=dest.slug,
            connector_id_map=connector_id_map,
        )

    return result
