from __future__ import annotations

from evenor.adapters.mail.body_format import (
    email_draft_html_from_markdown,
    format_email_bodies,
    html_to_plain,
    markdown_to_html,
    markdown_to_html_fragment,
    markdown_to_plain,
    normalize_email_draft_html,
    normalize_markdown_lists,
    normalize_signature_bullet_lines,
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


def test_normalize_signature_bullet_lines_strips_decorative_bullets() -> None:
    src = "Name\n\n* **Sales Executive**\n- **VDtec Distributors Ltd**"
    normalized = normalize_signature_bullet_lines(src)
    assert "**Sales Executive**" in normalized.splitlines()
    assert "* **Sales Executive**" not in normalized
    assert "- **VDtec Distributors Ltd**" not in normalized


def test_email_draft_flattens_signature_bullet_markdown() -> None:
    src = """Catherine Halftermeyer

* **Sales Executive**
* **VDtec Distributors Ltd**
Office 101, Ebene Junction"""
    html = email_draft_html_from_markdown(src)
    assert "<ul>" not in html
    assert "<strong>Sales Executive</strong>" in html
    assert "<strong>VDtec Distributors Ltd</strong>" in html
    assert "Office 101" in html


def test_email_draft_preserves_quote_list_with_text_after_bold() -> None:
    src = "* **SF 300 Clockers:** 22 units x 8,500 Rs"
    html = email_draft_html_from_markdown(src)
    assert "<ul>" in html
    assert "<li>" in html
    assert "22 units" in html


def test_normalize_email_draft_html_trims_strong_whitespace() -> None:
    html = normalize_email_draft_html("<p><strong> VDtec Distributors Ltd</strong></p>")
    assert html == "<p><strong>VDtec Distributors Ltd</strong></p>"


def test_normalize_email_draft_html_flattens_bold_only_ul() -> None:
    html = normalize_email_draft_html(
        "<ul><li><strong>Sales Executive</strong></li>"
        "<li><strong>VDtec Distributors Ltd</strong></li></ul>"
    )
    assert "<ul>" not in html
    assert "<p><strong>Sales Executive</strong></p>" in html
    assert "<p><strong>VDtec Distributors Ltd</strong></p>" in html


def test_normalize_email_draft_html_flattens_bold_only_ol() -> None:
    html = normalize_email_draft_html(
        "<ol><li><strong>Sales Executive</strong></li>"
        "<li><strong>VDtec Distributors Ltd</strong></li></ol>"
    )
    assert "<ol>" not in html
    assert "<p><strong>Sales Executive</strong></p>" in html


def test_normalize_email_draft_html_unwraps_decorative_blockquote() -> None:
    html = normalize_email_draft_html(
        "<p>Name</p><blockquote><p><strong>Sales Executive</strong><br>"
        "<strong>VDtec Distributors Ltd</strong></p></blockquote>"
    )
    assert "<blockquote>" not in html
    assert "<strong>Sales Executive</strong>" in html


def test_email_draft_flattens_blockquote_signature_markdown() -> None:
    src = """Catherine Halftermeyer

> **Sales Executive**
> **VDtec Distributors Ltd**
Office 101"""
    html = email_draft_html_from_markdown(src)
    assert "<blockquote>" not in html
    assert "<strong>Sales Executive</strong>" in html


def test_email_draft_flattens_numbered_bold_signature_markdown() -> None:
    src = """Catherine Halftermeyer

1. **Sales Executive**
2. **VDtec Distributors Ltd**"""
    html = email_draft_html_from_markdown(src)
    assert "<ol>" not in html
    assert "<p><strong>Sales Executive</strong></p>" in html


def test_normalize_email_draft_html_splits_br_separated_signature_block() -> None:
    html = normalize_email_draft_html(
        "<p>Catherine Halftermeyer</p>"
        "<p><strong>Sales Executive</strong><br>"
        "<strong>VDtec Distributors Ltd</strong><br>"
        "Office 101, Ebene Junction</p>"
    )
    assert "<p><strong>Sales Executive</strong></p>" in html
    assert "<p><strong>VDtec Distributors Ltd</strong></p>" in html
    assert "<p>Office 101, Ebene Junction</p>" in html
    assert "<br>" not in html


def test_prepare_email_draft_html_for_editor_normalizes_stored_ul() -> None:
    from evenor.adapters.mail.body_format import prepare_email_draft_html_for_editor

    html = prepare_email_draft_html_for_editor(
        "<ul><li><strong>Sales Executive</strong></li></ul>"
    )
    assert "<ul>" not in html
    assert "<strong>Sales Executive</strong>" in html
