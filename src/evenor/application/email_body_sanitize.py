from __future__ import annotations

import re
from html import unescape

from evenor.adapters.mail.body_format import _compact_plain_text, html_to_plain

_OUTLOOK_VML_LINE = re.compile(
    r"^\s*(?:v\\:\*|o\\:\*|w\\:\*|\.\s*shape)\s*\{[^}]*\}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OUTLOOK_VML_INLINE = re.compile(
    r"behavior\s*:\s*url\s*\(\s*#default#VML\s*\)",
    re.IGNORECASE,
)
_BLANK_LINES = re.compile(r"\n{3,}")
_HTML_MARKER = re.compile(r"<\s*(?:html|head|body|!doctype)\b", re.IGNORECASE)


def looks_like_html(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    if _HTML_MARKER.search(sample[:500]):
        return True
    tag_count = len(re.findall(r"<[^>]+>", sample))
    return tag_count >= 3 and tag_count * 10 > len(sample)


def normalize_inbound_body_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if looks_like_html(text):
        return html_to_plain(text)
    return text


def sanitize_plain(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = unescape(raw)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _OUTLOOK_VML_LINE.sub("", cleaned)
    cleaned = _OUTLOOK_VML_INLINE.sub("", cleaned)
    cleaned = _compact_plain_text(cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def prepare_email_body_new(body_text: str) -> str:
    from evenor.application.email_reply_parser import parse_reply_body

    normalized = normalize_inbound_body_text(body_text)
    sanitized = sanitize_plain(normalized)
    parsed = parse_reply_body(sanitized)
    return parsed.new_text or sanitized or normalized or body_text.strip()
