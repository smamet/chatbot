from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from chatbot.application.channel_outbound import persist_validation_email_subject
from chatbot.application.draft_edit_service import DraftEditError, save_pending_reply_draft
from chatbot.application.mail_imap_seen_service import mark_imap_seen_for_pending_reply
from chatbot.application.quote_pdf_storage import cleanup_pending_reply_attachments
from chatbot.application.validation_audit_service import ValidationAuditService
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import ConnectorType
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus


@dataclass(frozen=True, slots=True)
class BulkSkipItem:
    reply_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class BulkRejectResult:
    rejected_ids: list[int] = field(default_factory=list)
    skipped: list[BulkSkipItem] = field(default_factory=list)


def _persist_validation_email_draft_from_form(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    form: Mapping[str, Any],
    edited_by: str,
    outbound_config: dict | None = None,
) -> PendingReply:
    if reply.channel != ConnectorType.EMAIL.value:
        return reply
    if "draft_html" in form:
        try:
            reply = save_pending_reply_draft(
                session,
                tenant_id=tenant_id,
                reply=reply,
                draft_html=str(form.get("draft_html", "")),
                draft_subject=str(form.get("draft_subject", "")),
                edited_by=edited_by,
            )
        except DraftEditError:
            pass
    if outbound_config is not None:
        reply = persist_validation_email_subject(
            session,
            tenant_id=tenant_id,
            reply=reply,
            form_subject=str(form.get("draft_subject", "")),
            outbound_config=outbound_config,
        )
    return reply


def reject_pending_reply(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    edited_by: str,
    settings: Settings,
    form: Mapping[str, Any] | None = None,
    outbound_config: dict | None = None,
) -> None:
    form_data = form if form is not None else {}
    _persist_validation_email_draft_from_form(
        session,
        tenant_id=tenant_id,
        reply=reply,
        form=form_data,
        edited_by=edited_by,
        outbound_config=outbound_config,
    )
    cleanup_pending_reply_attachments(reply)
    mark_imap_seen_for_pending_reply(
        session,
        tenant_id=tenant_id,
        reply=reply,
        settings=settings,
    )
    ValidationAuditService(session).resolve_reply(
        reply,
        status=PendingReplyStatus.REJECTED,
        actor_email=edited_by,
    )


def format_bulk_reject_summary(result: BulkRejectResult) -> str:
    parts: list[str] = []
    count = len(result.rejected_ids)
    if count:
        label = "email" if count == 1 else "emails"
        parts.append(f"Rejected {count} {label}.")
    if result.skipped:
        skip_parts = [f"#{item.reply_id} ({item.reason})" for item in result.skipped]
        parts.append(f"Skipped {len(result.skipped)}: {', '.join(skip_parts)}.")
    return " ".join(parts)


class ValidationBulkService:
    def __init__(self, session: Session, *, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    def reject_email_replies(
        self,
        tenant_id: int,
        reply_ids: list[int],
        *,
        actor_email: str,
    ) -> BulkRejectResult:
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        repo = SqlAlchemyPendingReplyRepository(self._session)
        seen: set[int] = set()
        rejected_ids: list[int] = []
        skipped: list[BulkSkipItem] = []

        for raw_id in reply_ids:
            try:
                reply_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if reply_id in seen:
                continue
            seen.add(reply_id)

            reply = repo.find_by_id(reply_id)
            if reply is None:
                skipped.append(BulkSkipItem(reply_id=reply_id, reason="not found"))
                continue
            if reply.tenant_id != tenant_id:
                skipped.append(BulkSkipItem(reply_id=reply_id, reason="not found"))
                continue
            if reply.status != PendingReplyStatus.PENDING:
                skipped.append(BulkSkipItem(reply_id=reply_id, reason="not pending"))
                continue
            if reply.channel != ConnectorType.EMAIL.value:
                skipped.append(BulkSkipItem(reply_id=reply_id, reason="not email"))
                continue

            reject_pending_reply(
                self._session,
                tenant_id=tenant_id,
                reply=reply,
                edited_by=actor_email,
                settings=self._settings,
            )
            rejected_ids.append(reply_id)

        return BulkRejectResult(rejected_ids=rejected_ids, skipped=skipped)
