from __future__ import annotations

from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.tenant_service import TenantService
from chatbot.domain.contracts.vector_store import VectorRecord
from chatbot.domain.models.message import ChatMessage, MessageRole


def test_messages_isolated_by_tenant(test_settings) -> None:
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        svc = TenantService(SqlAlchemyTenantRepository(session))
        a = svc.create_tenant(name="A", slug="tenant-a")
        b = svc.create_tenant(name="B", slug="tenant-b")
        repo_a = SqlAlchemyConversationRepository(session, a.tenant.id)
        repo_b = SqlAlchemyConversationRepository(session, b.tenant.id)
        repo_a.append_message("shared-session", ChatMessage(role=MessageRole.USER, content="secret-a"))
        repo_b.append_message("shared-session", ChatMessage(role=MessageRole.USER, content="secret-b"))
        session.commit()
        msgs_a = repo_a.list_messages("shared-session")
        msgs_b = repo_b.list_messages("shared-session")
    finally:
        session.close()
        engine.dispose()
    assert len(msgs_a) == 1
    assert msgs_a[0].content == "secret-a"
    assert len(msgs_b) == 1
    assert msgs_b[0].content == "secret-b"


def test_lancedb_isolated_per_tenant_slug(test_settings) -> None:
    store_a = LanceVectorStore(test_settings.lancedb_root / "tenant-a")
    store_b = LanceVectorStore(test_settings.lancedb_root / "tenant-b")
    store_a.upsert(
        [
            VectorRecord(
                chunk_id="a1",
                text="only in A",
                source_path="/a/doc.md",
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )
    hits_b = store_b.search([1.0, 0.0, 0.0], top_k=5)
    hits_a = store_a.search([1.0, 0.0, 0.0], top_k=5)
    assert len(hits_a) == 1
    assert hits_a[0].text == "only in A"
    assert hits_b == []
