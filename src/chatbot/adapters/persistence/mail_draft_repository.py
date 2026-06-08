from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import MailDraftRow
from chatbot.domain.models.mail_draft import MailDraft, MailDraftStatus


def _row_to_draft(row: MailDraftRow) -> MailDraft:
    return MailDraft(
        id=row.id,
        tenant_id=row.tenant_id,
        imap_uid=row.imap_uid,
        from_addr=row.from_addr,
        to_addr=row.to_addr,
        subject=row.subject,
        body_in=row.body_in,
        draft_reply=row.draft_reply,
        status=MailDraftStatus(row.status),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
    )


class SqlAlchemyMailDraftRepository:
    def __init__(self, session: Session, *, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def find_by_reply(self, from_addr: str, draft_reply: str) -> MailDraft | None:
        row = self._session.scalar(
            select(MailDraftRow)
            .where(
                MailDraftRow.tenant_id == self._tenant_id,
                MailDraftRow.from_addr == from_addr.strip().lower(),
                MailDraftRow.draft_reply == draft_reply,
            )
            .order_by(MailDraftRow.id.desc())
            .limit(1)
        )
        return _row_to_draft(row) if row else None

    def exists_by_uid(self, imap_uid: str) -> bool:
        row = self._session.scalar(
            select(MailDraftRow.id).where(
                MailDraftRow.tenant_id == self._tenant_id,
                MailDraftRow.imap_uid == imap_uid,
            )
        )
        return row is not None

    def create(
        self,
        *,
        imap_uid: str,
        from_addr: str,
        to_addr: str,
        subject: str,
        body_in: str,
        status: MailDraftStatus = MailDraftStatus.PENDING,
    ) -> MailDraft:
        now = datetime.now(UTC)
        row = MailDraftRow(
            tenant_id=self._tenant_id,
            imap_uid=imap_uid,
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body_in=body_in,
            draft_reply="",
            status=status.value,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_draft(row)

    def mark_processed(self, draft_id: int, *, draft_reply: str) -> None:
        row = self._session.get(MailDraftRow, draft_id)
        if row is None or row.tenant_id != self._tenant_id:
            return
        row.draft_reply = draft_reply
        row.status = MailDraftStatus.PROCESSED.value
        row.updated_at = datetime.now(UTC)
        self._session.flush()

    def mark_failed(self, draft_id: int, *, error: str = "") -> None:
        row = self._session.get(MailDraftRow, draft_id)
        if row is None or row.tenant_id != self._tenant_id:
            return
        row.draft_reply = error[:2000] if error else row.draft_reply
        row.status = MailDraftStatus.FAILED.value
        row.updated_at = datetime.now(UTC)
        self._session.flush()
