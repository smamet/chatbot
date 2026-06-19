from __future__ import annotations

from datetime import UTC

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.domain.models.pending_reply import PendingReply


def inbound_subject_for_pending_reply(
    session: Session,
    tenant_id: int,
    *,
    channel: str,
    recipient_id: str,
    session_id: str,
    draft_text: str,
) -> str:
    channel_key = (channel or "").lower()
    if channel_key == "email":
        from_addr = (recipient_id or session_id.removeprefix("email:")).strip().lower()
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
        from_addr = (reply.recipient_id or reply.session_id.removeprefix("email:")).strip().lower()
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_by_reply(
            from_addr, reply.draft_text
        )
        if draft:
            return {
                "subject": draft.subject,
                "text": draft.body_in,
                "received_at": draft.created_at,
            }
    conv = SqlAlchemyConversationRepository(session, tenant_id)
    before = reply.created_at
    if before.tzinfo is None:
        before = before.replace(tzinfo=UTC)
    user_meta = conv.last_user_message_with_time_before(reply.session_id, before)
    if user_meta:
        text, received_at = user_meta
    return {"subject": subject, "text": text, "received_at": received_at}
