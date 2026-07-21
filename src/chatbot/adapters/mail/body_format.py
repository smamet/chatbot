from __future__ import annotations

import re
from html.parser import HTMLParser

import bleach
import markdown
from bleach.css_sanitizer import CSSSanitizer

from chatbot.adapters.channels.text_format import format_for_messenger

_EMAIL_WRAPPER = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: sans-serif; line-height: 1.5; color: #333;">
{body}
</body>
</html>"""

_MARKDOWN_EXTENSIONS = ["extra", "nl2br", "sane_lists"]
_BULLET_LINE_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+\.\s+)")
_SIGNATURE_BULLET_BOLD_RE = re.compile(r"^\s*[-*+]\s+(\*\*.+?\*\*)\s*$")
_SIGNATURE_BLOCKQUOTE_BOLD_RE = re.compile(r"^\s*>\s+(\*\*.+?\*\*)\s*$")

_EMAIL_HTML_TAGS = [
    "p",
    "br",
    "strong",
    "b",
    "em",
    "i",
    "u",
    "a",
    "ul",
    "ol",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "blockquote",
    "div",
    "span",
]
_EMAIL_HTML_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "span": ["style"],
    "p": ["style"],
}
_CSS_SANITIZER = CSSSanitizer(allowed_css_properties=["color", "background-color"])


def _compact_plain_text(text: str) -> str:
    """Drop whitespace-only lines left over from HTML block/tag conversion."""
    lines = [line.strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_signature_bullet_lines(text: str) -> str:
    """Drop bullet/blockquote markers from lines that are only a bold span (common in email signatures)."""
    out: list[str] = []
    for line in text.split("\n"):
        match = _SIGNATURE_BULLET_BOLD_RE.match(line) or _SIGNATURE_BLOCKQUOTE_BOLD_RE.match(line)
        if match:
            out.append(match.group(1))
        else:
            out.append(line)
    return "\n".join(out)


def normalize_markdown_lists(text: str) -> str:
    """Insert a blank line before list blocks when Python-Markdown would miss them."""
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        if (
            i > 0
            and _BULLET_LINE_RE.match(line)
            and lines[i - 1].strip()
            and not _BULLET_LINE_RE.match(lines[i - 1])
        ):
            if out and out[-1].strip():
                out.append("")
        out.append(line)
    return "\n".join(out)


def sanitize_email_html(html: str) -> str:
    cleaned = bleach.clean(
        html,
        tags=_EMAIL_HTML_TAGS,
        attributes=_EMAIL_HTML_ATTRS,
        css_sanitizer=_CSS_SANITIZER,
        strip=True,
    )
    return cleaned.strip()


def _trim_phrasing_whitespace(html: str) -> str:
    html = re.sub(r"<(strong|em|b|i)>\s+", r"<\1>", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+</(strong|em|b|i)>", r"</\1>", html, flags=re.IGNORECASE)
    html = re.sub(r"<p>\s+", "<p>", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+</p>", "</p>", html, flags=re.IGNORECASE)
    return html


def _li_bold_only_inner(html: str) -> str | None:
    inner = html.strip()
    if not inner:
        return None
    without_br = re.sub(r"<br\s*/?>", "", inner, flags=re.IGNORECASE).strip()
    match = re.fullmatch(
        r"<(?:strong|b)>(.*?)</(?:strong|b)>",
        without_br,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip()


def _flatten_bold_only_lists(html: str) -> str:
    def replace_list(match: re.Match[str]) -> str:
        list_body = match.group(1)
        li_inners = re.findall(r"<li[^>]*>(.*?)</li>", list_body, flags=re.IGNORECASE | re.DOTALL)
        if not li_inners:
            return match.group(0)
        paragraphs: list[str] = []
        for li_inner in li_inners:
            bold_text = _li_bold_only_inner(li_inner)
            if bold_text is None:
                return match.group(0)
            paragraphs.append(f"<p><strong>{bold_text}</strong></p>")
        return "".join(paragraphs)

    html = re.sub(r"<ul[^>]*>(.*?)</ul>", replace_list, html, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"<ol[^>]*>(.*?)</ol>", replace_list, html, flags=re.IGNORECASE | re.DOTALL)


def _unwrap_decorative_blockquotes(html: str) -> str:
    def replace_blockquote(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if re.search(r"<(?:ul|ol)\b", inner, flags=re.IGNORECASE):
            return match.group(0)
        return inner

    return re.sub(
        r"<blockquote[^>]*>(.*?)</blockquote>",
        replace_blockquote,
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _split_br_separated_paragraphs(html: str) -> str:
    def split_p(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = re.split(r"<br\s*/?>", inner, flags=re.IGNORECASE)
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) <= 1:
            return match.group(0)
        return "".join(f"<p>{part}</p>" for part in parts)

    return re.sub(r"<p>(.*?)</p>", split_p, html, flags=re.IGNORECASE | re.DOTALL)


def normalize_email_draft_html(html: str) -> str:
    """Trim stray whitespace and flatten decorative bold-only bullet lists for WYSIWYG display."""
    if not html:
        return ""
    normalized = _trim_phrasing_whitespace(html)
    normalized = _flatten_bold_only_lists(normalized)
    normalized = _unwrap_decorative_blockquotes(normalized)
    return _split_br_separated_paragraphs(normalized)


def prepare_email_draft_html_for_editor(html: str) -> str:
    """Sanitize and normalize stored HTML (migration/backfill; not used on dashboard load)."""
    return normalize_email_draft_html(sanitize_email_html(html))


def markdown_to_plain(text: str) -> str:
    """Strip markdown syntax for the text/plain email part."""
    return format_for_messenger(text)


def markdown_to_html_fragment(text: str) -> str:
    """Render markdown to an HTML fragment (no email wrapper)."""
    normalized = normalize_signature_bullet_lines(text)
    normalized = normalize_markdown_lists(normalized)
    return markdown.markdown(
        normalized,
        extensions=_MARKDOWN_EXTENSIONS,
        output_format="html5",
    )


def markdown_to_html(text: str) -> str:
    """Render markdown to a simple HTML document suitable for email clients."""
    return _EMAIL_WRAPPER.format(body=markdown_to_html_fragment(text))


class _HtmlPlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._list_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"style", "script"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "blockquote", "tr"}:
            if self._parts and not self._parts[-1].endswith("\n"):
                self._parts.append("\n")
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("• ")
            self._list_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"style", "script"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "li":
            self._list_depth = max(0, self._list_depth - 1)
            self._parts.append("\n")
        elif tag in {"ul", "ol"} and self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def get_text(self) -> str:
        return _compact_plain_text("".join(self._parts))


def html_to_plain(html: str) -> str:
    parser = _HtmlPlainTextParser()
    parser.feed(html)
    return parser.get_text()


def format_email_bodies(
    text: str,
    *,
    html_fragment: str | None = None,
) -> tuple[str, str]:
    """Return (plain_text, html) parts for a multipart email."""
    if html_fragment:
        plain = html_to_plain(html_fragment)
        return plain, _EMAIL_WRAPPER.format(body=html_fragment)
    return markdown_to_plain(text), markdown_to_html(text)


def email_draft_html_from_markdown(text: str) -> str:
    """Build sanitized HTML for validation WYSIWYG from LLM markdown."""
    html = markdown_to_html_fragment(text)
    cleaned = sanitize_email_html(html)
    return normalize_email_draft_html(cleaned)
