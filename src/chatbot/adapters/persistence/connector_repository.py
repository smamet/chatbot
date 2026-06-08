from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import ConnectorRow
from chatbot.adapters.persistence.secrets import decrypt_json, encrypt_json
from chatbot.domain.models.connector import (
    Connector,
    ConnectorDirection,
    ConnectorMode,
    ConnectorType,
)


def _row_to_connector(row: ConnectorRow) -> Connector:
    return Connector(
        id=row.id,
        tenant_id=row.tenant_id,
        direction=ConnectorDirection(row.direction),
        type=ConnectorType(row.type),
        mode=ConnectorMode(row.mode),
        config=decrypt_json(row.config_enc),
        active=bool(row.active),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
    )


class SqlAlchemyConnectorRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_tenant(self, tenant_id: int) -> list[Connector]:
        rows = self._session.scalars(
            select(ConnectorRow)
            .where(ConnectorRow.tenant_id == tenant_id)
            .order_by(ConnectorRow.direction, ConnectorRow.type)
        ).all()
        return [_row_to_connector(r) for r in rows]

    def find_by_id(self, connector_id: int) -> Connector | None:
        row = self._session.get(ConnectorRow, connector_id)
        return _row_to_connector(row) if row else None

    def find_by_tenant_direction_type(
        self,
        tenant_id: int,
        *,
        direction: ConnectorDirection,
        type: ConnectorType,
    ) -> Connector | None:
        row = self._session.scalar(
            select(ConnectorRow).where(
                ConnectorRow.tenant_id == tenant_id,
                ConnectorRow.direction == direction.value,
                ConnectorRow.type == type.value,
            )
        )
        return _row_to_connector(row) if row else None

    def list_active_by_type(
        self,
        *,
        direction: ConnectorDirection,
        type: ConnectorType,
    ) -> list[Connector]:
        rows = self._session.scalars(
            select(ConnectorRow).where(
                ConnectorRow.direction == direction.value,
                ConnectorRow.type == type.value,
                ConnectorRow.active.is_(True),
            )
        ).all()
        return [_row_to_connector(r) for r in rows]

    def find_active(
        self,
        tenant_id: int,
        *,
        direction: ConnectorDirection,
        type: ConnectorType,
    ) -> Connector | None:
        row = self._session.scalar(
            select(ConnectorRow).where(
                ConnectorRow.tenant_id == tenant_id,
                ConnectorRow.direction == direction.value,
                ConnectorRow.type == type.value,
                ConnectorRow.active.is_(True),
            )
        )
        return _row_to_connector(row) if row else None

    def delete(self, connector_id: int) -> bool:
        row = self._session.get(ConnectorRow, connector_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def create(
        self,
        *,
        tenant_id: int,
        direction: ConnectorDirection,
        type: ConnectorType,
        mode: ConnectorMode,
        config: dict,
        active: bool = True,
    ) -> Connector:
        now = datetime.now(UTC)
        row = ConnectorRow(
            tenant_id=tenant_id,
            direction=direction.value,
            type=type.value,
            mode=mode.value,
            config_enc=encrypt_json(config),
            active=active,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_connector(row)

    def update(
        self,
        connector_id: int,
        *,
        config: dict | None = None,
        active: bool | None = None,
        mode: ConnectorMode | None = None,
    ) -> Connector | None:
        row = self._session.get(ConnectorRow, connector_id)
        if row is None:
            return None
        if config is not None:
            row.config_enc = encrypt_json(config)
        if active is not None:
            row.active = active
        if mode is not None:
            row.mode = mode.value
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_connector(row)

    def get(self, connector_id: int) -> Connector | None:
        row = self._session.get(ConnectorRow, connector_id)
        return _row_to_connector(row) if row else None
