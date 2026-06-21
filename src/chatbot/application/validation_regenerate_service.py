from __future__ import annotations

from datetime import UTC

from markdownify import markdownify as html_to_markdown
from sqlalchemy.orm import Session

from chatbot.adapters.mail.body_format import email_draft_html_from_markdown, sanitize_email_html
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.pending_reply_edit_repository import (
    SqlAlchemyPendingReplyEditRepository,
)
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.chat_service import ChatService
from chatbot.application.draft_edit_service import draft_edit_text_diff
from chatbot.application.validation_audit_service import ValidationAuditService
from chatbot.config.settings import Settings
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.message import MessageRole
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus
from chatbot.domain.models.pending_reply_audit import ValidationAuditAction
from chatbot.domain.models.tenant import Tenant


class ValidationRegenerateError(RuntimeError):
    pass


def regenerate_pending_reply_from_raw(
    session: Session,
    tenant: Tenant,
    reply: PendingReply,
    *,
    settings: Settings,
    chat: ChatService,
    edited_by: str,
) -> PendingReply:
    if reply.status != PendingReplyStatus.PENDING:
        raise ValidationRegenerateError("Reply is not pending")
    if (reply.channel or "").lower() != "email":
        raise ValidationRegenerateError("Regenerate is only available for email replies")
    if reply.fulfillment_kind == FulfillmentKind.ERPNEXT_QUOTE:
        raise ValidationRegenerateError("Regenerate is not available for quote replies")

    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant.id)
    draft = draft_repo.find_for_pending_reply(reply)
    if draft is None:
        raise ValidationRegenerateError("No inbound mail draft linked to this reply")
    body_in = (draft.body_in or "").strip()
    if not body_in:
        raise ValidationRegenerateError("Raw inbound email body is empty")

    conv = SqlAlchemyConversationRepository(session, tenant.id)
    before = reply.created_at
    if before.tzinfo is None:
        before = before.replace(tzinfo=UTC)
    messages = conv.list_messages_before(reply.session_id, before)
    pending_user_content: str | None = None
    old_assistant_content = reply.draft_text or ""
    if len(messages) >= 2 and messages[-1].role == MessageRole.ASSISTANT:
        if messages[-2].role == MessageRole.USER:
            pending_user_content = messages[-2].content
        messages = messages[:-2]
    elif messages and messages[-1].role == MessageRole.ASSISTANT:
        messages = messages[:-1]

    result = chat.regenerate_assistant_reply(
        reply.session_id,
        history=messages,
        inbound_text=body_in,
    )
    new_markdown = (result.text or "").strip()
    new_html = sanitize_email_html(email_draft_html_from_markdown(new_markdown))
    before_html = reply.draft_html or sanitize_email_html(
        email_draft_html_from_markdown(old_assistant_content)
    )

    pending_repo = SqlAlchemyPendingReplyRepository(session)
    updated = pending_repo.update_draft(
        reply.id,
        draft_text=new_markdown,
        draft_html=new_html,
    )
    if updated is None:
        raise ValidationRegenerateError("Failed to update pending reply draft")

    conv.update_assistant_message_content(
        reply.session_id,
        old_content=old_assistant_content,
        new_content=new_markdown,
    )
    if pending_user_content is not None:
        conv.update_user_message_content(
            reply.session_id,
            old_content=pending_user_content,
            new_content=body_in,
            before=before,
        )

    diff = draft_edit_text_diff(before_html, new_html)
    if diff:
        SqlAlchemyPendingReplyEditRepository(session).create(
            tenant_id=tenant.id,
            pending_reply_id=reply.id,
            edited_by=edited_by,
            body_before=before_html,
            body_after=new_html,
            diff=diff,
        )

    ValidationAuditService(session).log_event(
        tenant_id=tenant.id,
        pending_reply_id=reply.id,
        action=ValidationAuditAction.REGENERATED,
        actor_email=edited_by,
    )
    return updated
