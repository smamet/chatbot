from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import (
    ConnectorRow,
    HookEventRow,
    IngestedFileRow,
    MailDraftRow,
    MessageRow,
    OrderEventRow,
    OrderItemRow,
    OrderRow,
    PendingReplyRow,
    TenantRow,
    UserBotAccessRow,
)
from chatbot.adapters.persistence.secrets import decrypt_text, encrypt_text
from chatbot.domain.models.tenant import BotType, Tenant, TenantConfig, TraderSettings


def _parse_bot_type(raw: str | None) -> BotType:
    try:
        return BotType(str(raw or BotType.ASSISTANT.value).strip().lower())
    except ValueError:
        return BotType.ASSISTANT


def _serialize_config(config: TenantConfig) -> str:
    """Persist config_json with fundmanager_token encrypted at rest."""
    data = json.loads(config.to_json())
    trader = dict(data.get("trader") or {})
    token = str(trader.pop("fundmanager_token", "") or "")
    trader.pop("fundmanager_token_enc", None)
    if token:
        trader["fundmanager_token_enc"] = encrypt_text(token)
    data["trader"] = trader
    return json.dumps(data, ensure_ascii=True)


def _deserialize_config(raw: str | None) -> TenantConfig:
    if not raw or not str(raw).strip():
        return TenantConfig()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return TenantConfig()
    if not isinstance(data, dict):
        return TenantConfig()
    trader = dict(data.get("trader") or {}) if isinstance(data.get("trader"), dict) else {}
    enc = trader.pop("fundmanager_token_enc", None)
    if enc and not trader.get("fundmanager_token"):
        trader["fundmanager_token"] = decrypt_text(str(enc)) or ""
    data["trader"] = trader
    return TenantConfig.from_json(json.dumps(data, ensure_ascii=True))


def _row_to_tenant(row: TenantRow) -> Tenant:
    return Tenant(
        id=row.id,
        slug=row.slug,
        name=row.name,
        prompt=row.prompt or "",
        hook_instructions=row.hook_instructions,
        gemini_api_key=decrypt_text(row.gemini_api_key_enc) or None,
        config=_deserialize_config(row.config_json),
        active=bool(row.active),
        bot_type=_parse_bot_type(getattr(row, "bot_type", None)),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
        client_billing_input_per_million_usd=row.client_billing_input_per_million_usd,
        client_billing_output_per_million_usd=row.client_billing_output_per_million_usd,
    )


class SqlAlchemyTenantRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_id(self, tenant_id: int) -> Tenant | None:
        row = self._session.get(TenantRow, tenant_id)
        return _row_to_tenant(row) if row else None

    def find_by_slug(self, slug: str) -> Tenant | None:
        row = self._session.scalar(select(TenantRow).where(TenantRow.slug == slug))
        return _row_to_tenant(row) if row else None

    def find_by_token_hash(self, token_hash: str) -> Tenant | None:
        row = self._session.scalar(select(TenantRow).where(TenantRow.token_hash == token_hash))
        return _row_to_tenant(row) if row else None

    def list_all(self) -> list[Tenant]:
        rows = self._session.scalars(select(TenantRow).order_by(TenantRow.slug)).all()
        return [_row_to_tenant(r) for r in rows]

    def list_active_traders(self) -> list[Tenant]:
        rows = self._session.scalars(
            select(TenantRow)
            .where(TenantRow.active.is_(True), TenantRow.bot_type == BotType.TRADER.value)
            .order_by(TenantRow.slug)
        ).all()
        return [_row_to_tenant(r) for r in rows]

    def create(
        self,
        *,
        slug: str,
        name: str,
        token_hash: str,
        prompt: str,
        config: TenantConfig,
        hook_instructions: str | None = None,
        gemini_api_key: str | None = None,
        bot_type: BotType | str = BotType.ASSISTANT,
    ) -> Tenant:
        now = datetime.now(UTC)
        parsed = _parse_bot_type(bot_type.value if isinstance(bot_type, BotType) else bot_type)
        row = TenantRow(
            slug=slug,
            name=name,
            token_hash=token_hash,
            prompt=prompt,
            hook_instructions=hook_instructions,
            gemini_api_key_enc=encrypt_text(gemini_api_key) if gemini_api_key else None,
            config_json=_serialize_config(config),
            active=True,
            bot_type=parsed.value,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_tenant(row)

    def update(
        self,
        tenant_id: int,
        *,
        name: str | None = None,
        prompt: str | None = None,
        hook_instructions: str | None = None,
        update_hook_instructions: bool = False,
        config: TenantConfig | None = None,
        active: bool | None = None,
        token_hash: str | None = None,
        gemini_api_key: str | None = None,
        update_gemini_api_key: bool = False,
        bot_type: BotType | str | None = None,
        client_billing_input_per_million_usd: Decimal | None = None,
        client_billing_output_per_million_usd: Decimal | None = None,
        update_client_billing: bool = False,
    ) -> Tenant | None:
        row = self._session.get(TenantRow, tenant_id)
        if row is None:
            return None
        if name is not None:
            row.name = name
        if prompt is not None:
            row.prompt = prompt
        if update_hook_instructions:
            row.hook_instructions = hook_instructions
        if config is not None:
            row.config_json = _serialize_config(config)
        if active is not None:
            row.active = active
        if token_hash is not None:
            row.token_hash = token_hash
        if bot_type is not None:
            row.bot_type = _parse_bot_type(
                bot_type.value if isinstance(bot_type, BotType) else bot_type
            ).value
        if update_gemini_api_key:
            row.gemini_api_key_enc = encrypt_text(gemini_api_key) if gemini_api_key else None
        if update_client_billing:
            row.client_billing_input_per_million_usd = client_billing_input_per_million_usd
            row.client_billing_output_per_million_usd = client_billing_output_per_million_usd
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_tenant(row)

    def delete(self, tenant_id: int) -> bool:
        row = self._session.get(TenantRow, tenant_id)
        if row is None:
            return False
        order_ids = list(
            self._session.scalars(select(OrderRow.id).where(OrderRow.tenant_id == tenant_id))
        )
        if order_ids:
            self._session.execute(delete(OrderItemRow).where(OrderItemRow.order_id.in_(order_ids)))
            self._session.execute(delete(OrderEventRow).where(OrderEventRow.tenant_id == tenant_id))
            self._session.execute(delete(OrderRow).where(OrderRow.tenant_id == tenant_id))
        self._session.execute(delete(MessageRow).where(MessageRow.tenant_id == tenant_id))
        self._session.execute(delete(IngestedFileRow).where(IngestedFileRow.tenant_id == tenant_id))
        self._session.execute(delete(HookEventRow).where(HookEventRow.tenant_id == tenant_id))
        self._session.execute(delete(PendingReplyRow).where(PendingReplyRow.tenant_id == tenant_id))
        self._session.execute(delete(ConnectorRow).where(ConnectorRow.tenant_id == tenant_id))
        self._session.execute(delete(MailDraftRow).where(MailDraftRow.tenant_id == tenant_id))
        self._session.execute(delete(UserBotAccessRow).where(UserBotAccessRow.tenant_id == tenant_id))
        self._session.delete(row)
        self._session.flush()
        return True


# Re-export for type checkers / callers that previously imported only TenantConfig helpers.
__all__ = [
    "SqlAlchemyTenantRepository",
    "TraderSettings",
    "_deserialize_config",
    "_serialize_config",
]
