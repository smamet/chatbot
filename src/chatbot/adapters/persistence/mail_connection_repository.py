from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import MailConnectionRow
from chatbot.adapters.persistence.secrets import decrypt_json, encrypt_json
from chatbot.domain.models.mail_connection import MailConnection, MailConnectionProvider


def _row_to_connection(row: MailConnectionRow) -> MailConnection:
    return MailConnection(
        id=row.id,
        tenant_id=row.tenant_id,
        label=row.label,
        provider=MailConnectionProvider(row.provider),
        mailbox_email=row.mailbox_email,
        config=decrypt_json(row.config_enc),
        active=bool(row.active),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
    )


class SqlAlchemyMailConnectionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_tenant(self, tenant_id: int) -> list[MailConnection]:
        rows = self._session.scalars(
            select(MailConnectionRow)
            .where(MailConnectionRow.tenant_id == tenant_id)
            .order_by(MailConnectionRow.label, MailConnectionRow.id)
        ).all()
        return [_row_to_connection(r) for r in rows]

    def find_by_id(self, connection_id: int) -> MailConnection | None:
        row = self._session.get(MailConnectionRow, connection_id)
        return _row_to_connection(row) if row else None

    def find_for_tenant(self, connection_id: int, tenant_id: int) -> MailConnection | None:
        row = self._session.scalar(
            select(MailConnectionRow).where(
                MailConnectionRow.id == connection_id,
                MailConnectionRow.tenant_id == tenant_id,
            )
        )
        return _row_to_connection(row) if row else None

    def create(
        self,
        *,
        tenant_id: int,
        label: str,
        provider: MailConnectionProvider,
        mailbox_email: str,
        config: dict,
        active: bool = True,
    ) -> MailConnection:
        now = datetime.now(UTC)
        row = MailConnectionRow(
            tenant_id=tenant_id,
            label=label,
            provider=provider.value,
            mailbox_email=mailbox_email,
            config_enc=encrypt_json(config),
            active=active,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_connection(row)

    def update(
        self,
        connection_id: int,
        *,
        label: str | None = None,
        mailbox_email: str | None = None,
        config: dict | None = None,
        active: bool | None = None,
    ) -> MailConnection | None:
        row = self._session.get(MailConnectionRow, connection_id)
        if row is None:
            return None
        if label is not None:
            row.label = label
        if mailbox_email is not None:
            row.mailbox_email = mailbox_email
        if config is not None:
            row.config_enc = encrypt_json(config)
        if active is not None:
            row.active = active
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_connection(row)

    def delete(self, connection_id: int) -> bool:
        row = self._session.get(MailConnectionRow, connection_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True
