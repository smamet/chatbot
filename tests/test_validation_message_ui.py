from __future__ import annotations

from datetime import UTC, datetime

from chatbot.application.validation_message_ui import bubble_from_content, bubble_from_draft
from chatbot.domain.models.mail_draft import MailDraft, MailDraftStatus
from chatbot.domain.models.message import MessageRole

_NOW = datetime.now(tz=UTC)


def test_bubble_hides_sanitize_meta_when_clean_equals_raw() -> None:
    bubble = bubble_from_draft(
        draft=MailDraft(
            id=1,
            tenant_id=1,
            imap_uid="1",
            from_addr="a@b.com",
            to_addr="bot@test.local",
            subject="Hi",
            body_in="Same text",
            body_new="Same text",
            draft_reply="",
            status=MailDraftStatus.PENDING,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    assert bubble.content_clean == "Same text"
    assert bubble.content_raw is None
    assert bubble.token_raw is None
    assert bubble.reduction_pct is None


def test_bubble_shows_sanitize_meta_when_clean_differs_from_raw() -> None:
    bubble = bubble_from_content(
        role=MessageRole.USER,
        content="clean",
        content_raw="raw with extra",
    )
    assert bubble.content_raw == "raw with extra"
    assert bubble.token_raw is not None
    assert bubble.reduction_pct is not None


def test_bubble_from_content_converts_html_legacy_message() -> None:
    html = "<html><body><p>Legacy quarantine text</p></body></html>"
    bubble = bubble_from_content(role=MessageRole.USER, content=html)
    assert bubble.content_clean == "Legacy quarantine text"
    assert bubble.content_raw == html
    assert bubble.token_raw is not None


def test_email_from_addr_from_session() -> None:
    from chatbot.application.validation_message_ui import email_from_addr_from_session

    assert email_from_addr_from_session("email:client@example.com~abc123") == "client@example.com"
    assert email_from_addr_from_session("whatsapp:+123") is None
