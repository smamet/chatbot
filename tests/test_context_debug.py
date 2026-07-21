from __future__ import annotations

from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.application.context_debug import (
    context_debug_from_json,
    context_debug_to_json,
    format_context_debug_label,
)
from chatbot.domain.models.context_debug import ContextDebugInfo
from chatbot.domain.models.message import ChatMessage, MessageRole


def test_context_debug_json_roundtrip() -> None:
    debug = ContextDebugInfo(rag_chunks=5, rag_chars=12340, customer_chars=840, system_chars=18500)
    raw = context_debug_to_json(debug)
    assert raw is not None
    restored = context_debug_from_json(raw)
    assert restored == debug


def test_format_context_debug_label() -> None:
    label = format_context_debug_label(
        ContextDebugInfo(rag_chunks=3, rag_chars=1200, customer_chars=0, system_chars=5000)
    )
    assert "RAG: 3 chunks" in label
    assert "1.2k chars" in label
    assert "Customer:" not in label
    assert "System: 5.0k chars" in label


def test_assistant_message_persists_context_debug(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    debug = ContextDebugInfo(rag_chunks=2, rag_chars=500, customer_chars=100, system_chars=900)
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        repo.append_message(
            "sess-debug",
            ChatMessage(role=MessageRole.ASSISTANT, content="reply", context_debug=debug),
        )
        session.commit()
        msgs = repo.list_messages("sess-debug", limit=10)
    finally:
        session.close()
    assert len(msgs) == 1
    assert msgs[0].context_debug == debug
