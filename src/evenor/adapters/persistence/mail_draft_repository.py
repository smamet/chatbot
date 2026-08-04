from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from evenor.adapters.persistence.orm import MailDraftRow
from evenor.domain.models.mail_draft import MailDraft, MailDraftStatus
from evenor.domain.models.pending_reply import PendingReply


def _row_to_draft(row: MailDraftRow) -> MailDraft:
    return MailDraft(
        id=row.id,
        tenant_id=row.tenant_id,
        imap_uid=row.imap_uid,
        from_addr=row.from_addr,
        to_addr=row.to_addr,
        subject=row.subject,
        body_in=row.body_in,
        body_new=row.body_new or "",
        draft_reply=row.draft_reply,
        status=MailDraftStatus(row.status),
        thread_id=row.thread_id,
        message_id=row.message_id or "",
        in_reply_to=row.in_reply_to or "",
        references_header=row.references_header,
        normalized_subject=row.normalized_subject or "",
        thread_resolution_json=row.thread_resolution_json,
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
    )


class SqlAlchemyMailDraftRepository:
    def __init__(self, session: Session, *, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def find_by_id(self, draft_id: int) -> MailDraft | None:
        row = self._session.get(MailDraftRow, draft_id)
        if row is None or row.tenant_id != self._tenant_id:
            return None
        return _row_to_draft(row)

    def find_by_message_id(self, message_id: str) -> MailDraft | None:
        mid = message_id.strip()
        if not mid:
            return None
        row = self._session.scalar(
            select(MailDraftRow)
            .where(
                MailDraftRow.tenant_id == self._tenant_id,
                MailDraftRow.message_id == mid,
            )
            .order_by(MailDraftRow.id.desc())
            .limit(1)
        )
        return _row_to_draft(row) if row else None

    def find_latest_for_thread(self, thread_id: int) -> MailDraft | None:
        row = self._session.scalar(
            select(MailDraftRow)
            .where(
                MailDraftRow.tenant_id == self._tenant_id,
                MailDraftRow.thread_id == thread_id,
            )
            .order_by(MailDraftRow.id.desc())
            .limit(1)
        )
        return _row_to_draft(row) if row else None

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

    def find_for_pending_reply(self, reply: PendingReply) -> MailDraft | None:
        if reply.mail_draft_id is not None:
            draft = self.find_by_id(reply.mail_draft_id)
            if draft is not None:
                return draft
        from_addr = (reply.recipient_id or reply.session_id.removeprefix("email:")).strip().lower()
        if "~" in from_addr:
            from_addr = from_addr.split("~", 1)[0]
        draft = self.find_by_reply(from_addr, reply.draft_text)
        if draft is not None:
            return draft
        before = reply.created_at
        if before.tzinfo is None:
            before = before.replace(tzinfo=UTC)
        row = self._session.scalar(
            select(MailDraftRow)
            .where(
                MailDraftRow.tenant_id == self._tenant_id,
                MailDraftRow.from_addr == from_addr,
                MailDraftRow.imap_uid != "",
                MailDraftRow.created_at <= before,
            )
            .order_by(MailDraftRow.id.desc())
            .limit(1)
        )
        return _row_to_draft(row) if row else None

    def find_nearest_before(
        self,
        *,
        thread_id: int | None,
        from_addr: str,
        before: datetime,
    ) -> MailDraft | None:
        if before.tzinfo is None:
            before = before.replace(tzinfo=UTC)
        stmt = select(MailDraftRow).where(
            MailDraftRow.tenant_id == self._tenant_id,
            MailDraftRow.from_addr == from_addr.strip().lower(),
            MailDraftRow.created_at <= before,
        )
        if thread_id is not None:
            stmt = stmt.where(MailDraftRow.thread_id == thread_id)
        row = self._session.scalar(stmt.order_by(MailDraftRow.created_at.desc()).limit(1))
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
        body_new: str = "",
        status: MailDraftStatus = MailDraftStatus.PENDING,
        thread_id: int | None = None,
        message_id: str = "",
        in_reply_to: str = "",
        references_header: str | None = None,
        normalized_subject: str = "",
        thread_resolution_json: str | None = None,
    ) -> MailDraft:
        now = datetime.now(UTC)
        row = MailDraftRow(
            tenant_id=self._tenant_id,
            imap_uid=imap_uid,
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body_in=body_in,
            body_new=body_new or body_in,
            draft_reply="",
            status=status.value,
            thread_id=thread_id,
            message_id=message_id,
            in_reply_to=in_reply_to,
            references_header=references_header,
            normalized_subject=normalized_subject,
            thread_resolution_json=thread_resolution_json,
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
