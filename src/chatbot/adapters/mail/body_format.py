from __future__ import annotations

import markdown

from chatbot.adapters.channels.text_format import format_for_messenger

_EMAIL_WRAPPER = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: sans-serif; line-height: 1.5; color: #333;">
{body}
</body>
</html>"""


def markdown_to_plain(text: str) -> str:
    """Strip markdown syntax for the text/plain email part."""
    return format_for_messenger(text)


def markdown_to_html(text: str) -> str:
    """Render markdown to a simple HTML document suitable for email clients."""
    rendered = markdown.markdown(
        text,
        extensions=["extra", "nl2br", "sane_lists"],
        output_format="html5",
    )
    return _EMAIL_WRAPPER.format(body=rendered)


def format_email_bodies(text: str) -> tuple[str, str]:
    """Return (plain_text, html) parts for a multipart email."""
    return markdown_to_plain(text), markdown_to_html(text)
