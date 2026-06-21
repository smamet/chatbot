from __future__ import annotations

from chatbot.adapters.mail.body_format import (
    format_email_bodies,
    html_to_plain,
    markdown_to_html,
    markdown_to_html_fragment,
    markdown_to_plain,
    normalize_markdown_lists,
)


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


def test_normalize_markdown_lists_inserts_blank_line() -> None:
    src = "Here is the list:\n* Item one\n* Item two"
    normalized = normalize_markdown_lists(src)
    assert normalized.split("\n")[1] == ""


def test_markdown_list_after_paragraph_renders_ul() -> None:
    src = "Here is the list:\n* Item one\n* Item two"
    html = markdown_to_html_fragment(src)
    assert "<ul>" in html
    assert "<li>" in html


def test_format_email_bodies_uses_html_fragment_when_provided() -> None:
    fragment = "<p>Hello <strong>world</strong></p><ul><li>One</li></ul>"
    plain, html = format_email_bodies("ignored", html_fragment=fragment)
    assert "world" in plain
    assert "• One" in plain
    assert fragment in html
    assert "<!DOCTYPE html>" in html


def test_html_to_plain_converts_lists() -> None:
    plain = html_to_plain("<p>Hi</p><ul><li>Alpha</li><li>Beta</li></ul>")
    assert "Alpha" in plain
    assert "Beta" in plain


def test_html_to_plain_compacts_nested_block_gaps() -> None:
    html = (
        "<html><body><div><div><p>Line one</p></div>"
        "<div><p>Line two</p></div></div></body></html>"
    )
    plain = html_to_plain(html)
    assert plain == "Line one\nLine two"
    assert "\n\n\n" not in plain
