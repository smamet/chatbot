from __future__ import annotations

from sqlalchemy import select

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.orm import PendingReplyEditRow
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.draft_edit_service import save_pending_reply_draft
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.message import ChatMessage, MessageRole


def test_save_pending_reply_draft_logs_diff_and_syncs_message(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="**Hello** client",
            draft_html="<p><strong>Hello</strong> client</p>",
        )
        conv = SqlAlchemyConversationRepository(session, tenant.id)
        conv.append_message(
            "email:client@example.com",
            ChatMessage(role=MessageRole.ASSISTANT, content="**Hello** client"),
        )
        session.commit()
        pending_id = pending.id

    with factory() as session:
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(pending_id)
        assert reply is not None
        updated = save_pending_reply_draft(
            session,
            tenant_id=tenant.id,
            reply=reply,
            draft_html="<p>Hello <em>edited</em> client</p>",
            edited_by="admin@example.com",
        )
        session.commit()

    assert updated.draft_html is not None
    assert "<em>edited</em>" in updated.draft_html
    assert "edited" in updated.draft_text

    with factory() as session:
        messages = SqlAlchemyConversationRepository(session, tenant.id).list_messages(
            "email:client@example.com"
        )
        assert len(messages) == 1
        assert "edited" in messages[0].content
        edits = session.scalars(select(PendingReplyEditRow)).all()
        assert len(edits) == 1
        assert edits[0].edited_by == "admin@example.com"
        assert edits[0].diff
