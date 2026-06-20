from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chatbot.application.email_body_sanitize import prepare_email_body_new
from chatbot.application.email_text_tokens import estimate_text_tokens, reduction_percent
from chatbot.domain.models.mail_draft import MailDraft
from chatbot.domain.models.message import MessageRole


@dataclass(frozen=True, slots=True)
class ValidationMessageBubble:
    role: MessageRole
    content_clean: str
    content_raw: str | None
    token_raw: int | None
    token_new: int
    reduction_pct: int | None
    created_at: datetime | None = None


def clean_body_for_display(body_in: str, body_new: str = "") -> str:
    if (body_new or "").strip():
        return body_new.strip()
    return prepare_email_body_new(body_in)


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
