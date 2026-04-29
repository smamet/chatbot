from __future__ import annotations

import re


_BULLET_RE = re.compile(r"^\s*[-*]\s+", flags=re.MULTILINE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", flags=re.MULTILINE)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _common_cleanup(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _MULTISPACE_RE.sub(" ", normalized)
    normalized = _MANY_BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def _strip_markdown_tokens(text: str) -> str:
    cleaned = _HEADING_RE.sub("", text)
    cleaned = _BULLET_RE.sub("• ", cleaned)
    cleaned = re.sub(r"[*_`~]", "", cleaned)
    return _common_cleanup(cleaned)


def format_for_messenger(text: str) -> str:
    return _strip_markdown_tokens(text)


def format_for_instagram(text: str) -> str:
    return _strip_markdown_tokens(text)


def format_for_whatsapp(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove malformed emphasis patterns such as "* *Title".
    cleaned = re.sub(r"(^|\n)\s*\*\s+\*", r"\1* ", cleaned)
    cleaned = _MANY_BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()
