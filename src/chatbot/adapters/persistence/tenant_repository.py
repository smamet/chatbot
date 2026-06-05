from __future__ import annotations

from datetime import UTC, datetime

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
    TenantRow,
    UserBotAccessRow,
)
from chatbot.adapters.persistence.secrets import decrypt_text, encrypt_text
from chatbot.domain.models.tenant import Tenant, TenantConfig


def _row_to_tenant(row: TenantRow) -> Tenant:
    return Tenant(
        id=row.id,
        slug=row.slug,
        name=row.name,
        prompt=row.prompt or "",
        hook_instructions=row.hook_instructions,
        gemini_api_key=decrypt_text(row.gemini_api_key_enc) or None,
        config=TenantConfig.from_json(row.config_json),
        active=bool(row.active),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
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
    ) -> Tenant:
        now = datetime.now(UTC)
        row = TenantRow(
            slug=slug,
            name=name,
            token_hash=token_hash,
            prompt=prompt,
            hook_instructions=hook_instructions,
            gemini_api_key_enc=encrypt_text(gemini_api_key) if gemini_api_key else None,
            config_json=config.to_json(),
            active=True,
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
            row.config_json = config.to_json()
        if active is not None:
            row.active = active
        if token_hash is not None:
            row.token_hash = token_hash
        if update_gemini_api_key:
            row.gemini_api_key_enc = encrypt_text(gemini_api_key) if gemini_api_key else None
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
        self._session.execute(delete(ConnectorRow).where(ConnectorRow.tenant_id == tenant_id))
        self._session.execute(delete(MailDraftRow).where(MailDraftRow.tenant_id == tenant_id))
        self._session.execute(delete(UserBotAccessRow).where(UserBotAccessRow.tenant_id == tenant_id))
        self._session.delete(row)
        self._session.flush()
        return True
