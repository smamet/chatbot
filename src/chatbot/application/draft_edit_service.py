from __future__ import annotations

import difflib
import re
from html.parser import HTMLParser

from markdownify import markdownify as html_to_markdown
from sqlalchemy.orm import Session

from chatbot.adapters.mail.body_format import normalize_email_draft_html, sanitize_email_html
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.pending_reply_edit_repository import (
    SqlAlchemyPendingReplyEditRepository,
)
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus


class DraftEditError(RuntimeError):
    pass


_BLOCK_END_TAGS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "blockquote", "div"})


class _HtmlBlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[str] = []
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self._flush()
        elif tag.lower() == "li":
            self._buf.append("• ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _BLOCK_END_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        self._buf.append(data)

    def _flush(self) -> None:
        text = "".join(self._buf)
        text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
        self._buf = []
        if text:
            self.blocks.append(text)


def _normalize_diff_block(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def _html_blocks_for_diff(html: str) -> list[str]:
    parser = _HtmlBlockParser()
    parser.feed(html or "")
    parser.close()
    return parser.blocks


def draft_edit_text_diff(before_html: str, after_html: str) -> str:
    before_blocks = _html_blocks_for_diff(before_html)
    after_blocks = _html_blocks_for_diff(after_html)
    if [_normalize_diff_block(block) for block in before_blocks] == [
        _normalize_diff_block(block) for block in after_blocks
    ]:
        return ""
    lines = list(
        difflib.unified_diff(
            before_blocks,
            after_blocks,
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def submitted_draft_matches_stored(
    reply: PendingReply,
    draft_html: str,
    draft_subject: str | None = None,
) -> bool:
    sanitized = normalize_email_draft_html(sanitize_email_html(draft_html))
    if sanitized != (reply.draft_html or ""):
        return False
    if draft_subject is not None:
        if draft_subject.strip() != (reply.draft_subject or "").strip():
            return False
    return True


def save_pending_reply_draft(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    draft_html: str,
    edited_by: str,
    draft_subject: str | None = None,
) -> PendingReply:
    if reply.status != PendingReplyStatus.PENDING:
        raise DraftEditError("Reply is not pending")

    sanitized = normalize_email_draft_html(sanitize_email_html(draft_html))
    before_html = reply.draft_html or ""
    subject_changed = False
    sanitized_subject: str | None = None
    if draft_subject is not None:
        sanitized_subject = draft_subject.strip()
        if sanitized_subject != (reply.draft_subject or "").strip():
            subject_changed = True

    if sanitized == before_html and not subject_changed:
        return reply

    markdown_text = html_to_markdown(sanitized, heading_style="ATX").strip()
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    old_draft_text = reply.draft_text

    updated = pending_repo.update_draft(
        reply.id,
        draft_html=sanitized if sanitized != before_html else None,
        draft_text=markdown_text if sanitized != before_html else None,
        draft_subject=sanitized_subject if subject_changed else None,
    )
    if updated is None:
        raise DraftEditError("Failed to update draft")

    if sanitized != before_html:
        diff = draft_edit_text_diff(before_html, sanitized)
        if subject_changed:
            diff = (
                f"Subject: {reply.draft_subject or ''!r} -> {sanitized_subject!r}\n" + diff
            )
        SqlAlchemyPendingReplyEditRepository(session).create(
            tenant_id=tenant_id,
            pending_reply_id=reply.id,
            edited_by=edited_by,
            body_before=before_html,
            body_after=sanitized,
            diff=diff,
        )

        conv = SqlAlchemyConversationRepository(session, tenant_id)
        conv.update_assistant_message_content(
            reply.session_id,
            old_content=old_draft_text,
            new_content=markdown_text,
        )
    elif subject_changed:
        SqlAlchemyPendingReplyEditRepository(session).create(
            tenant_id=tenant_id,
            pending_reply_id=reply.id,
            edited_by=edited_by,
            body_before=before_html,
            body_after=before_html,
            diff=f"Subject: {reply.draft_subject or ''!r} -> {sanitized_subject!r}\n",
        )
    return updated
