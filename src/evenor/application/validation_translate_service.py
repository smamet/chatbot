from __future__ import annotations

import json
import re

from sqlalchemy.orm import Session

from evenor.adapters.llm.gemini_client import GeminiLlmClient
from evenor.adapters.mail.body_format import normalize_email_draft_html, sanitize_email_html
from evenor.application.tenant_settings import merge_tenant_settings
from evenor.application.usage_metering import metered_llm
from evenor.config.settings import Settings
from evenor.domain.contracts.llm_client import LlmClient
from evenor.domain.models.message import ChatMessage, MessageRole
from evenor.domain.models.pending_reply import PendingReply, PendingReplyStatus
from evenor.domain.models.tenant import Tenant

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class ValidationTranslateError(RuntimeError):
    pass


def _gemini_api_key(tenant: Tenant, settings: Settings) -> str:
    return (tenant.gemini_api_key or settings.gemini_api_key or "").strip()


def _parse_translate_response(raw: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        raise ValidationTranslateError("Empty translation response")
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationTranslateError("Invalid translation response") from exc
    if not isinstance(data, dict):
        raise ValidationTranslateError("Invalid translation response")
    draft_html = data.get("draft_html")
    if not isinstance(draft_html, str) or not draft_html.strip():
        raise ValidationTranslateError("Translation response missing draft_html")
    draft_subject = data.get("draft_subject")
    if draft_subject is not None and not isinstance(draft_subject, str):
        raise ValidationTranslateError("Translation response has invalid draft_subject")
    return {
        "draft_html": draft_html.strip(),
        "draft_subject": (draft_subject or "").strip(),
    }


def _translate_system_instruction(target_lang: str) -> str:
    lang_label = "English" if target_lang == "en" else "French"
    return (
        f"You translate outbound business email drafts to {lang_label}. "
        "Reply with JSON only: {\"draft_html\": \"...\", \"draft_subject\": \"...\"}. "
        "Preserve HTML structure (<p>, <br>, lists). "
        "Do not translate proper names, email addresses, URLs, phone numbers, "
        "product codes, or company names. "
        "Translate the subject line when provided; use an empty string for draft_subject if none was given."
    )


def translate_pending_reply_draft(
    reply: PendingReply,
    *,
    draft_html: str,
    draft_subject: str | None,
    target_lang: str,
    tenant: Tenant,
    settings: Settings,
    session: Session,
    llm: LlmClient | None = None,
) -> dict[str, str]:
    if reply.status != PendingReplyStatus.PENDING:
        raise ValidationTranslateError("Reply is not pending")
    if (reply.channel or "").lower() != "email":
        raise ValidationTranslateError("Translation is only available for email replies")

    lang = (target_lang or "").strip().lower()
    if lang not in ("en", "fr"):
        raise ValidationTranslateError("target_lang must be en or fr")

    html_in = (draft_html or "").strip()
    if not html_in:
        raise ValidationTranslateError("Draft body is empty")

    subject_in = (draft_subject or "").strip()
    user_lines = [f"Target language: {lang}", f"Subject: {subject_in or '(none)'}", "", html_in]
    user_content = "\n".join(user_lines)

    if llm is None:
        merged = merge_tenant_settings(settings, tenant)
        api_key = _gemini_api_key(tenant, settings) or None
        llm = metered_llm(
            inner=GeminiLlmClient(model=merged.rewrite_model, api_key=api_key),
            tenant_id=tenant.id,
            operation="translate",
            model=merged.rewrite_model,
            session=session,
        )

    result = llm.generate_chat(
        system_instruction=_translate_system_instruction(lang),
        messages=[ChatMessage(role=MessageRole.USER, content=user_content)],
    )
    parsed = _parse_translate_response(result.text)
    sanitized_html = normalize_email_draft_html(sanitize_email_html(parsed["draft_html"]))
    if not sanitized_html.strip():
        raise ValidationTranslateError("Translation produced empty body")
    out_subject = parsed["draft_subject"] or subject_in
    return {"draft_html": sanitized_html, "draft_subject": out_subject}
