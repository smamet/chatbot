from __future__ import annotations

import difflib

from markdownify import markdownify as html_to_markdown
from sqlalchemy.orm import Session

from chatbot.adapters.mail.body_format import sanitize_email_html
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.pending_reply_edit_repository import (
    SqlAlchemyPendingReplyEditRepository,
)
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus


class DraftEditError(RuntimeError):
    pass


def _html_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )


def save_pending_reply_draft(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    draft_html: str,
    edited_by: str,
) -> PendingReply:
    if reply.status != PendingReplyStatus.PENDING:
        raise DraftEditError("Reply is not pending")

    sanitized = sanitize_email_html(draft_html)
    before_html = reply.draft_html or ""
    if sanitized == before_html:
        return reply

    markdown_text = html_to_markdown(sanitized, heading_style="ATX").strip()
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    old_draft_text = reply.draft_text

    updated = pending_repo.update_draft(
        reply.id,
        draft_html=sanitized,
        draft_text=markdown_text,
    )
    if updated is None:
        raise DraftEditError("Failed to update draft")

    SqlAlchemyPendingReplyEditRepository(session).create(
        tenant_id=tenant_id,
        pending_reply_id=reply.id,
        edited_by=edited_by,
        body_before=before_html,
        body_after=sanitized,
        diff=_html_diff(before_html, sanitized),
    )

    conv = SqlAlchemyConversationRepository(session, tenant_id)
    conv.update_assistant_message_content(
        reply.session_id,
        old_content=old_draft_text,
        new_content=markdown_text,
    )
    return updated
