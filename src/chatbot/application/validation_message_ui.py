from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.application.email_body_sanitize import looks_like_html, prepare_email_body_new
from chatbot.application.email_text_tokens import estimate_text_tokens, reduction_percent
from chatbot.domain.models.mail_draft import MailDraft
from chatbot.domain.models.message import ChatMessage, MessageRole


@dataclass(frozen=True, slots=True)
class ValidationMessageBubble:
    role: MessageRole
    content_clean: str
    content_raw: str | None
    token_raw: int | None
    token_new: int
    reduction_pct: int | None
    created_at: datetime | None = None


_CHANNEL_LABELS = {
    "email": "Email",
    "whatsapp": "WhatsApp",
    "instagram": "Instagram",
    "messenger": "Messenger",
}


def channel_label(channel: str | None) -> str:
    key = (channel or "").strip().lower()
    return _CHANNEL_LABELS.get(key, key or "Unknown")


def email_from_addr_from_session(session_id: str) -> str | None:
    if not session_id.startswith("email:"):
        return None
    from_addr = session_id.removeprefix("email:").strip().lower()
    if "~" in from_addr:
        from_addr = from_addr.split("~", 1)[0]
    return from_addr or None


def bubble_to_ui_dict(
    bubble: ValidationMessageBubble | None,
    *,
    role: MessageRole,
    content: str,
    context_debug: dict | None = None,
) -> dict:
    out: dict = {"role": role.value, "content": content}
    if bubble is not None:
        out.update(
            {
                "content_clean": bubble.content_clean,
                "content_raw": bubble.content_raw,
                "token_raw": bubble.token_raw,
                "token_new": bubble.token_new,
                "reduction_pct": bubble.reduction_pct,
                "show_sanitize_meta": bubble.content_raw is not None,
            }
        )
    elif role == MessageRole.ASSISTANT:
        out["content_clean"] = content
        out["markdown"] = True
    else:
        out["content_clean"] = content
    if context_debug:
        out["context_size"] = context_debug
    return out


def clean_body_for_display(body_in: str, body_new: str = "") -> str:
    if (body_in or "").strip():
        return prepare_email_body_new(body_in)
    if (body_new or "").strip():
        return prepare_email_body_new(body_new)
    return ""


def _sanitize_meta(
    content_raw: str | None,
    content_clean: str,
) -> tuple[str | None, int | None, int | None]:
    raw = (content_raw or "").strip()
    if not raw:
        return None, None, None
    if (content_clean or "").strip() == raw:
        return None, None, None
    token_raw = estimate_text_tokens(raw)
    token_new = estimate_text_tokens(content_clean)
    return content_raw, token_raw, reduction_percent(token_raw, token_new)


def bubble_from_draft(
    *,
    draft: MailDraft,
    role: MessageRole = MessageRole.USER,
    created_at: datetime | None = None,
) -> ValidationMessageBubble:
    content_clean = clean_body_for_display(draft.body_in, draft.body_new)
    content_raw, token_raw, reduction_pct = _sanitize_meta(draft.body_in or None, content_clean)
    token_new = estimate_text_tokens(content_clean)
    return ValidationMessageBubble(
        role=role,
        content_clean=content_clean,
        content_raw=content_raw,
        token_raw=token_raw,
        token_new=token_new,
        reduction_pct=reduction_pct,
        created_at=created_at or draft.created_at,
    )


def bubble_from_content(
    *,
    role: MessageRole,
    content: str,
    created_at: datetime | None = None,
    content_raw: str | None = None,
) -> ValidationMessageBubble:
    source = (content_raw or content or "").strip()
    if looks_like_html(source):
        content_clean = prepare_email_body_new(source)
        content_raw, token_raw, reduction_pct = _sanitize_meta(source, content_clean)
    else:
        content_clean = (content or "").strip()
        content_raw, token_raw, reduction_pct = _sanitize_meta(content_raw, content_clean)
    token_new = estimate_text_tokens(content_clean)
    return ValidationMessageBubble(
        role=role,
        content_clean=content_clean,
        content_raw=content_raw,
        token_raw=token_raw,
        token_new=token_new,
        reduction_pct=reduction_pct,
        created_at=created_at,
    )


def bubble_for_session_message(
    session: Session,
    tenant_id: int,
    session_id: str,
    message: ChatMessage,
    *,
    thread_id: int | None = None,
) -> ValidationMessageBubble | None:
    if message.role == MessageRole.ASSISTANT:
        return None
    from_addr = email_from_addr_from_session(session_id)
    if from_addr is not None:
        created_at = message.created_at
        if created_at is not None and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        draft = draft_repo.find_nearest_before(
            thread_id=thread_id,
            from_addr=from_addr,
            before=created_at or datetime.now(tz=UTC),
        )
        if draft is not None:
            return bubble_from_draft(
                draft=draft,
                role=message.role,
                created_at=message.created_at,
            )
    return bubble_from_content(
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def messages_to_bubbles(
    session: Session,
    tenant_id: int,
    session_id: str,
    messages: list[ChatMessage],
    *,
    thread_id: int | None = None,
) -> list[tuple[ChatMessage, ValidationMessageBubble | None]]:
    return [
        (
            message,
            bubble_for_session_message(
                session,
                tenant_id,
                session_id,
                message,
                thread_id=thread_id,
            ),
        )
        for message in messages
    ]
