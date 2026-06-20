from __future__ import annotations

from chatbot.application.email_body_sanitize import prepare_email_body_new, sanitize_plain
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
