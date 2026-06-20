from __future__ import annotations

from chatbot.application.email_reply_parser import parse_reply_body


def test_parse_reply_body_gmail_quote() -> None:
    body = "Merci pour l'info.\n\nOn Mon, Jun 1, 2026 Alice wrote:\n> old text"
    parsed = parse_reply_body(body)
    assert parsed.new_text == "Merci pour l'info."
    assert "Alice wrote" in parsed.quoted_text


def test_parse_reply_body_french_quote() -> None:
    body = "OK pour le devis.\n\nLe 1 juin 2026 à 10:00, Bob a écrit :\n> ancien"
    parsed = parse_reply_body(body)
    assert parsed.new_text == "OK pour le devis."


def test_parse_reply_body_keeps_full_when_only_quote() -> None:
    body = "> quoted only"
    parsed = parse_reply_body(body)
    assert parsed.new_text == body
