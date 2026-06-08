from __future__ import annotations

from chatbot.adapters.mail.body_format import format_email_bodies, markdown_to_html, markdown_to_plain


def test_markdown_to_plain_strips_formatting() -> None:
    src = "Hello **world**\n\n1. First\n2. Second"
    plain = markdown_to_plain(src)
    assert "**" not in plain
    assert "world" in plain
    assert "1. First" in plain


def test_markdown_to_html_renders_bold_and_lists() -> None:
    src = "Hello **world**\n\n- Item one\n- Item two"
    html = markdown_to_html(src)
    assert "<strong>world</strong>" in html
    assert "<li>" in html
    assert "<!DOCTYPE html>" in html


def test_format_email_bodies_returns_plain_and_html() -> None:
    plain, html = format_email_bodies("**Hi** there")
    assert "**" not in plain
    assert "Hi" in plain
    assert "<strong>Hi</strong>" in html
