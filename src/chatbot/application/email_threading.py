from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.outbound_email_message_repository import (
    SqlAlchemyOutboundEmailMessageRepository,
)
from chatbot.application.email_message_id import build_references_header, normalize_message_id
from chatbot.domain.models.pending_reply import PendingReply


@dataclass(frozen=True, slots=True)
class EmailThreadingContext:
    in_reply_to: str | None
    references: str | None


def resolve_threading_for_reply(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
) -> EmailThreadingContext | None:
    thread_id = reply.thread_id
    if thread_id is None and reply.mail_draft_id is not None:
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_by_id(
            reply.mail_draft_id
        )
        if draft is not None:
            thread_id = draft.thread_id

    if thread_id is None:
        return None

    outbound_repo = SqlAlchemyOutboundEmailMessageRepository(session, tenant_id=tenant_id)
    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)

    parent_mid: str | None = None
    refs_parts: list[str | None] = []

    latest_out = outbound_repo.find_latest_for_thread(thread_id)
    if latest_out is not None:
        parent_mid = latest_out.message_id
        refs_parts.append(latest_out.references_header)
        refs_parts.append(latest_out.message_id)
    else:
        latest_draft = draft_repo.find_latest_for_thread(thread_id)
        if latest_draft is not None and latest_draft.message_id:
            parent_mid = latest_draft.message_id
            refs_parts.append(latest_draft.references_header)
            refs_parts.append(latest_draft.message_id)

    if not parent_mid:
        return None

    in_reply_to = normalize_message_id(parent_mid)
    references = build_references_header(*refs_parts)
    return EmailThreadingContext(in_reply_to=in_reply_to, references=references)
