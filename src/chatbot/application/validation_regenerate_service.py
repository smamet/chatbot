from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

from sqlalchemy.orm import Session

from chatbot.adapters.mail.body_format import email_draft_html_from_markdown
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.application.chat_service import ChatService
from chatbot.config.settings import Settings
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus
from chatbot.domain.models.tenant import Tenant


class ValidationRegenerateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RegenerateDraftResult:
    draft_html: str
    draft_text: str


@dataclass(frozen=True, slots=True)
class _RegenerateContext:
    session_id: str
    body_in: str
    history: list[ChatMessage]


def _prepare_regenerate(
    session: Session,
    tenant: Tenant,
    reply: PendingReply,
) -> _RegenerateContext:
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
    if len(messages) >= 2 and messages[-1].role == MessageRole.ASSISTANT:
        if messages[-2].role == MessageRole.USER:
            pass
        messages = messages[:-2]
    elif messages and messages[-1].role == MessageRole.ASSISTANT:
        messages = messages[:-1]

    return _RegenerateContext(
        session_id=reply.session_id,
        body_in=body_in,
        history=messages,
    )


def generate_pending_reply_from_raw(
    session: Session,
    tenant: Tenant,
    reply: PendingReply,
    *,
    settings: Settings,
    chat: ChatService,
) -> RegenerateDraftResult:
    _ = settings
    ctx = _prepare_regenerate(session, tenant, reply)
    result = chat.regenerate_assistant_reply(
        ctx.session_id,
        history=ctx.history,
        inbound_text=ctx.body_in,
    )
    draft_text = (result.text or "").strip()
    draft_html = email_draft_html_from_markdown(draft_text)
    if not draft_html.strip():
        raise ValidationRegenerateError("Regenerate produced empty body")
    return RegenerateDraftResult(draft_html=draft_html, draft_text=draft_text)
