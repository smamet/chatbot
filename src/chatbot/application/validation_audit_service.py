from __future__ import annotations

import json

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.pending_reply_audit_repository import (
    SqlAlchemyPendingReplyAuditRepository,
)
from chatbot.adapters.persistence.pending_reply_edit_repository import (
    SqlAlchemyPendingReplyEditRepository,
)
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus
from chatbot.domain.models.pending_reply_audit import (
    PendingReplyAuditEvent,
    ValidationAuditAction,
    ValidationTimelineEntry,
)


def _audit_summary(action: ValidationAuditAction, detail_json: str | None) -> str:
    detail: dict = {}
    if detail_json:
        try:
            detail = json.loads(detail_json)
        except (TypeError, json.JSONDecodeError):
            pass
    if action == ValidationAuditAction.APPROVED:
        return "Approved reply"
    if action == ValidationAuditAction.REJECTED:
        return "Rejected reply"
    if action == ValidationAuditAction.ATTACHMENT_ADDED:
        name = detail.get("filename", "file")
        return f"Added attachment {name}"
    if action == ValidationAuditAction.ATTACHMENT_REMOVED:
        name = detail.get("filename", "file")
        return f"Removed attachment {name}"
    if action == ValidationAuditAction.RESOLVE_PRODUCTS:
        return "Resolved products"
    if action == ValidationAuditAction.REFRESH_PDF:
        quote = detail.get("quote_name")
        return f"Synced PDF{f' ({quote})' if quote else ''}"
    return action.value


def _merge_timeline(
    edits: list,
    audits: list[PendingReplyAuditEvent],
) -> list[ValidationTimelineEntry]:
    entries: list[ValidationTimelineEntry] = []
    for edit in edits:
        entries.append(
            ValidationTimelineEntry(
                kind="edit",
                pending_reply_id=edit.pending_reply_id,
                actor_email=edit.edited_by,
                created_at=edit.created_at,
                action="edit",
                summary="Edited draft",
                diff=edit.diff or None,
            )
        )
    for event in audits:
        entries.append(
            ValidationTimelineEntry(
                kind="audit",
                pending_reply_id=event.pending_reply_id,
                actor_email=event.actor_email,
                created_at=event.created_at,
                action=event.action.value,
                summary=_audit_summary(event.action, event.detail_json),
                detail_json=event.detail_json,
            )
        )
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


class ValidationAuditService:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._audit = SqlAlchemyPendingReplyAuditRepository(session)
        self._edits = SqlAlchemyPendingReplyEditRepository(session)
        self._pending = SqlAlchemyPendingReplyRepository(session)

    def log_event(
        self,
        *,
        tenant_id: int,
        pending_reply_id: int,
        action: ValidationAuditAction,
        actor_email: str,
        detail: dict | None = None,
    ) -> PendingReplyAuditEvent:
        return self._audit.create(
            tenant_id=tenant_id,
            pending_reply_id=pending_reply_id,
            action=action,
            actor_email=actor_email,
            detail=detail,
        )

    def resolve_reply(
        self,
        reply: PendingReply,
        *,
        status: PendingReplyStatus,
        actor_email: str,
    ) -> PendingReply | None:
        if status not in (PendingReplyStatus.APPROVED, PendingReplyStatus.REJECTED):
            raise ValueError("resolve_reply requires approved or rejected status")
        updated = self._pending.resolve(reply.id, status=status, resolved_by=actor_email)
        if updated is None:
            return None
        action = (
            ValidationAuditAction.APPROVED
            if status == PendingReplyStatus.APPROVED
            else ValidationAuditAction.REJECTED
        )
        self.log_event(
            tenant_id=reply.tenant_id,
            pending_reply_id=reply.id,
            action=action,
            actor_email=actor_email,
        )
        return updated

    def list_activity(self, tenant_id: int, *, limit: int = 50) -> list[ValidationTimelineEntry]:
        edits = self._edits.list_for_tenant(tenant_id, limit=limit)
        audits = self._audit.list_for_tenant(tenant_id, limit=limit)
        merged = _merge_timeline(edits, audits)
        return merged[:limit]

    def list_timeline_for_reply(
        self, tenant_id: int, pending_reply_id: int, *, limit: int = 100
    ) -> list[ValidationTimelineEntry]:
        edits = self._edits.list_for_reply(tenant_id, pending_reply_id, limit=limit)
        audits = self._audit.list_for_reply(tenant_id, pending_reply_id, limit=limit)
        return _merge_timeline(edits, audits)[:limit]
