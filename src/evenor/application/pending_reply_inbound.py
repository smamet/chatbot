from __future__ import annotations

from sqlalchemy.orm import Session

from evenor.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.domain.models.pending_reply import PendingReply
from evenor.application.validation_message_ui import clean_body_for_display


def inbound_subject_for_pending_reply(
    session: Session,
    tenant_id: int,
    *,
    channel: str,
    recipient_id: str,
    session_id: str,
    draft_text: str,
    mail_draft_id: int | None = None,
) -> str:
    channel_key = (channel or "").lower()
    if channel_key == "email":
        if mail_draft_id is not None:
            draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_by_id(
                mail_draft_id
            )
            if draft and draft.subject:
                return draft.subject.strip()
        from_addr = (recipient_id or session_id.removeprefix("email:")).strip().lower()
        if "~" in from_addr:
            from_addr = from_addr.split("~", 1)[0]
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_by_reply(
            from_addr, draft_text
        )
        if draft and draft.subject:
            return draft.subject.strip()
    return ""


def inbound_for_pending_reply(session: Session, tenant_id: int, reply: PendingReply) -> dict:
    subject = ""
    text = ""
    received_at = None
    channel = (reply.channel or "").lower()
    if channel == "email":
        draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        draft = None
        if reply.mail_draft_id is not None:
            draft = draft_repo.find_by_id(reply.mail_draft_id)
        if draft is None:
            from_addr = (reply.recipient_id or reply.session_id.removeprefix("email:")).strip().lower()
            if "~" in from_addr:
                from_addr = from_addr.split("~", 1)[0]
            draft = draft_repo.find_by_reply(from_addr, reply.draft_text)
        if draft:
            display_text = clean_body_for_display(draft.body_in, draft.body_new)
            return {
                "subject": draft.subject,
                "text": display_text,
                "raw_text": draft.body_in or "",
                "received_at": draft.created_at,
                "mail_draft": draft,
            }
    conv = SqlAlchemyConversationRepository(session, tenant_id)
    before = reply.created_at
    if before.tzinfo is None:
        from datetime import UTC

        before = before.replace(tzinfo=UTC)
    user_meta = conv.last_user_message_with_time_before(reply.session_id, before)
    if user_meta:
        text, received_at = user_meta
    return {"subject": subject, "text": text, "received_at": received_at}
