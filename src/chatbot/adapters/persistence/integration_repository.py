from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import IntegrationRow
from chatbot.adapters.persistence.secrets import decrypt_json, encrypt_json
from chatbot.domain.models.integration import Integration, IntegrationType


def _row_to_integration(row: IntegrationRow) -> Integration:
    return Integration(
        id=row.id,
        tenant_id=row.tenant_id,
        type=IntegrationType(row.type),
        config=decrypt_json(row.config_enc),
        active=bool(row.active),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
    )


class SqlAlchemyIntegrationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_tenant(self, tenant_id: int) -> list[Integration]:
        rows = self._session.scalars(
            select(IntegrationRow)
            .where(IntegrationRow.tenant_id == tenant_id)
            .order_by(IntegrationRow.type)
        ).all()
        return [_row_to_integration(r) for r in rows]

    def find_by_id(self, integration_id: int) -> Integration | None:
        row = self._session.get(IntegrationRow, integration_id)
        return _row_to_integration(row) if row else None

    def find_by_tenant_type(
        self,
        tenant_id: int,
        *,
        type: IntegrationType,
    ) -> Integration | None:
        row = self._session.scalar(
            select(IntegrationRow).where(
                IntegrationRow.tenant_id == tenant_id,
                IntegrationRow.type == type.value,
            )
        )
        return _row_to_integration(row) if row else None

    def find_active(self, tenant_id: int, *, type: IntegrationType) -> Integration | None:
        row = self._session.scalar(
            select(IntegrationRow).where(
                IntegrationRow.tenant_id == tenant_id,
                IntegrationRow.type == type.value,
                IntegrationRow.active.is_(True),
            )
        )
        return _row_to_integration(row) if row else None

    def delete(self, integration_id: int) -> bool:
        row = self._session.get(IntegrationRow, integration_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def create(
        self,
        *,
        tenant_id: int,
        type: IntegrationType,
        config: dict,
        active: bool = True,
    ) -> Integration:
        now = datetime.now(UTC)
        row = IntegrationRow(
            tenant_id=tenant_id,
            type=type.value,
            config_enc=encrypt_json(config),
            active=active,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_integration(row)

    def update(
        self,
        integration_id: int,
        *,
        config: dict | None = None,
        active: bool | None = None,
    ) -> Integration | None:
        row = self._session.get(IntegrationRow, integration_id)
        if row is None:
            return None
        if config is not None:
            row.config_enc = encrypt_json(config)
        if active is not None:
            row.active = active
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_integration(row)

    def get(self, integration_id: int) -> Integration | None:
        return self.find_by_id(integration_id)
