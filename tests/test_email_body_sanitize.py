from __future__ import annotations

from chatbot.application.email_body_sanitize import (
    looks_like_html,
    prepare_email_body_new,
    sanitize_plain,
)
from chatbot.application.email_text_tokens import estimate_text_tokens, reduction_percent


def test_sanitize_plain_unescapes_nbsp() -> None:
    assert sanitize_plain("Bonjour,&nbsp;monde") == "Bonjour, monde"


def test_sanitize_plain_strips_outlook_vml() -> None:
    body = (
        "v\\:* {behavior:url(#default#VML);}\n"
        "o\\:* {behavior:url(#default#VML);}\n"
        "Bonjour,\n"
        "J'espère que vous allez bien."
    )
    cleaned = sanitize_plain(body)
    assert "behavior:url" not in cleaned
    assert "Bonjour," in cleaned


def test_prepare_email_body_new_strips_quotes() -> None:
    body = "Merci.\n\nOn Mon wrote:\n> old"
    assert prepare_email_body_new(body) == "Merci."


def test_estimate_text_tokens() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("a" * 8) == 2


def test_reduction_percent() -> None:
    assert reduction_percent(100, 25) == 75
    assert reduction_percent(0, 10) is None


def test_looks_like_html_detects_m365_document() -> None:
    html = "<html><head><style>body{color:#000}</style></head><body><p>Hi</p></body></html>"
    assert looks_like_html(html)


def test_prepare_email_body_new_strips_html_and_style() -> None:
    html = (
        "<html><head><meta charset='utf-8'>"
        "<style>body, table, td { font-size: 14px; }</style></head>"
        "<body><p>Microsoft 365 security quarantine notice.</p>"
        "<p>Review your messages.</p></body></html>"
    )
    cleaned = prepare_email_body_new(html)
    assert "<html" not in cleaned.lower()
    assert "font-size" not in cleaned
    assert "Microsoft 365 security quarantine notice." in cleaned
    assert "Review your messages." in cleaned
    assert "\n\n\n" not in cleaned
