from __future__ import annotations

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.interfaces.api.routers.dashboard_web import _conversation_history_for_pending_reply


def test_conversation_history_excludes_current_draft(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session_id = "email:client@example.com"

    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        conv = SqlAlchemyConversationRepository(session, tenant.id)
        conv.append_message(session_id, ChatMessage(role=MessageRole.USER, content="First question"))
        conv.append_message(session_id, ChatMessage(role=MessageRole.ASSISTANT, content="First answer"))
        conv.append_message(session_id, ChatMessage(role=MessageRole.USER, content="Second question"))
        conv.append_message(session_id, ChatMessage(role=MessageRole.ASSISTANT, content="Current draft"))
        session.flush()
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id=session_id,
            channel="email",
            recipient_id="client@example.com",
            draft_text="Current draft",
        )
        session.commit()
        reply = pending

    with factory() as session:
        history = _conversation_history_for_pending_reply(session, tenant.id, reply)

    assert len(history) == 3
    assert history[0].content_clean == "First question"
    assert history[1].content_clean == "First answer"
    assert history[2].content_clean == "Second question"
    assert all(m.content_clean != "Current draft" for m in history)
    assert all(m.created_at is not None for m in history)
