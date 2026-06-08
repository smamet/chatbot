from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import MailImapUidRow


class SqlAlchemyMailImapUidRepository:
    def __init__(self, session: Session, *, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def exists_by_uid(self, imap_uid: str) -> bool:
        row = self._session.scalar(
            select(MailImapUidRow.id).where(
                MailImapUidRow.tenant_id == self._tenant_id,
                MailImapUidRow.imap_uid == imap_uid,
            )
        )
        return row is not None

    def record_skipped(
        self,
        imap_uid: str,
        *,
        received_at: datetime | None = None,
    ) -> None:
        if self.exists_by_uid(imap_uid):
            return
        row = MailImapUidRow(
            tenant_id=self._tenant_id,
            imap_uid=imap_uid,
            disposition="skipped",
            received_at=received_at,
            created_at=datetime.now(UTC),
        )
        self._session.add(row)
        self._session.flush()

    def record_processed(
        self,
        imap_uid: str,
        *,
        received_at: datetime | None = None,
    ) -> None:
        if self.exists_by_uid(imap_uid):
            return
        row = MailImapUidRow(
            tenant_id=self._tenant_id,
            imap_uid=imap_uid,
            disposition="processed",
            received_at=received_at,
            created_at=datetime.now(UTC),
        )
        self._session.add(row)
        self._session.flush()
