from __future__ import annotations

import uuid

from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.application.chat_service import ChatService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.domain.contracts.llm_client import LlmResult, LlmUsage
from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage, MessageRole
from dataclasses import replace

from chatbot.domain.models.tenant import TenantConfig


class FakeLlm:
    def __init__(self, reply: str = "hello") -> None:
        self._reply = reply

    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
        attachments: list[Attachment] | None = None,
    ) -> LlmResult:
        _ = system_instruction
        _ = messages
        _ = attachments
        return LlmResult(text=self._reply, usage=LlmUsage(prompt_tokens=1, candidates_tokens=2, total_tokens=3))


def test_chat_service_roundtrip(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    fake = FakeLlm("Thanks for your message.")
    sid = f"sess-{uuid.uuid4().hex}"
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(settings=test_settings, tenant=tenant, llm=fake, repo=repo, rag=None)
        out = svc.handle_user_message(sid, "Hi there")
        assert out.text == "Thanks for your message."
        assert out.usage.total_tokens == 3
        session.commit()
        msgs = repo.list_messages(sid, limit=10)
    finally:
        session.close()
    assert len(msgs) == 2
    assert msgs[0].role == MessageRole.USER
    assert msgs[0].content == "Hi there"
    assert msgs[1].role == MessageRole.ASSISTANT
    assert msgs[1].content == "Thanks for your message."


def test_chat_service_persists_attachment_notes_and_forwards_to_llm(
    test_settings, test_tenant
) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    class CaptureLlm(FakeLlm):
        def __init__(self) -> None:
            super().__init__("ok")
            self.last_attachments: list[Attachment] | None = None

        def generate_chat(
            self,
            *,
            system_instruction: str,
            messages: list[ChatMessage],
            attachments: list[Attachment] | None = None,
        ) -> LlmResult:
            _ = system_instruction
            _ = messages
            self.last_attachments = attachments
            return super().generate_chat(
                system_instruction=system_instruction,
                messages=messages,
                attachments=attachments,
            )

    cap = CaptureLlm()
    sid = f"sess-att-{uuid.uuid4().hex}"
    att = Attachment(
        mime_type="application/pdf",
        data=b"%PDF-1.4",
        filename="quote.pdf",
    )
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(settings=test_settings, tenant=tenant, llm=cap, repo=repo, rag=None)
        svc.handle_user_message(sid, "See attached", attachments=[att])
        session.commit()
        msgs = repo.list_messages(sid, limit=10)
    finally:
        session.close()

    assert msgs[0].content == "See attached\n[Attached: quote.pdf]"
    assert cap.last_attachments is not None
    assert len(cap.last_attachments) == 1
    assert cap.last_attachments[0].filename == "quote.pdf"


def test_chat_service_includes_rag_context(test_settings, test_tenant) -> None:
    from chatbot.domain.contracts.vector_store import RetrievedChunk, VectorRecord

    tenant, _token = test_tenant
    tenant = replace(
        tenant, config=TenantConfig(rag_enabled=True, rag_rewrite_enabled=False)
    )

    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 1.0] for _ in texts]

    class FakeStore:
        def delete_by_source_path(self, source_path: str) -> None:
            _ = source_path

        def upsert(self, records: list[VectorRecord]) -> None:
            _ = records

        def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
            _ = query_vector
            _ = top_k
            return [
                RetrievedChunk(
                    chunk_id="c1",
                    text="Widget price is 42 EUR",
                    source_path="/tmp/price.csv",
                    score=0.1,
                )
            ]

    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    class CaptureLlm:
        def __init__(self) -> None:
            self.last_system: str | None = None

        def generate_chat(
            self,
            *,
            system_instruction: str,
            messages: list[ChatMessage],
            attachments: list[Attachment] | None = None,
        ) -> LlmResult:
            _ = attachments
            self.last_system = system_instruction
            return LlmResult(text="ok", usage=LlmUsage())

    rewriter = FakeLlm("unused")
    rag = RagPipeline(
        settings=test_settings.model_copy(update={"rag_enabled": True, "rag_rewrite_enabled": False}),
        rewriter_llm=rewriter,
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
    )
    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(settings=test_settings, tenant=tenant, llm=cap, repo=repo, rag=rag)
        svc.handle_user_message("s2", "How much?")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "Retrieved context" in cap.last_system
    assert "42 EUR" in cap.last_system
    assert "/tmp/price.csv" not in cap.last_system
    assert "Do not mention internal file names" in cap.last_system


def test_chat_service_rag_includes_source_paths_when_dev_mode(test_settings, test_tenant) -> None:
    from chatbot.domain.contracts.vector_store import RetrievedChunk, VectorRecord

    tenant, _token = test_tenant
    tenant = replace(
        tenant,
        config=TenantConfig(rag_enabled=True, rag_rewrite_enabled=False, dev_mode=True),
    )

    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 1.0] for _ in texts]

    class FakeStore:
        def delete_by_source_path(self, source_path: str) -> None:
            _ = source_path

        def upsert(self, records: list[VectorRecord]) -> None:
            _ = records

        def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
            _ = query_vector
            _ = top_k
            return [
                RetrievedChunk(
                    chunk_id="c1",
                    text="Widget price is 42 EUR",
                    source_path="/tmp/price.csv",
                    score=0.1,
                )
            ]

    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    class CaptureLlm:
        def __init__(self) -> None:
            self.last_system: str | None = None

        def generate_chat(
            self,
            *,
            system_instruction: str,
            messages: list[ChatMessage],
            attachments: list[Attachment] | None = None,
        ) -> LlmResult:
            _ = attachments
            self.last_system = system_instruction
            return LlmResult(text="ok", usage=LlmUsage())

    rewriter = FakeLlm("unused")
    rag = RagPipeline(
        settings=test_settings.model_copy(update={"rag_enabled": True, "dev_mode": True}),
        rewriter_llm=rewriter,
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
    )
    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(settings=test_settings, tenant=tenant, llm=cap, repo=repo, rag=rag)
        svc.handle_user_message("s3", "How much?")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "/tmp/price.csv" in cap.last_system
    assert "chunk c1" in cap.last_system
    assert "Do not mention internal file names" not in cap.last_system


def test_chat_service_includes_quote_hook_when_erpnext_active(test_settings, test_tenant) -> None:
    from datetime import UTC, datetime

    tenant, _token = test_tenant
    tenant = replace(
        tenant,
        config=TenantConfig(automation_modules=("erpnext.quote",), rag_enabled=False),
        updated_at=datetime.now(UTC),
    )
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    class CaptureLlm:
        def __init__(self) -> None:
            self.last_system: str | None = None

        def generate_chat(
            self,
            *,
            system_instruction: str,
            messages: list[ChatMessage],
            attachments: list[Attachment] | None = None,
        ) -> LlmResult:
            _ = messages, attachments
            self.last_system = system_instruction
            return LlmResult(text="ok", usage=LlmUsage())

    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(
            settings=test_settings,
            tenant=tenant,
            llm=cap,
            repo=repo,
            rag=None,
            active_integrations={"erpnext"},
        )
        svc.handle_user_message("email:client@example.com", "Quote please")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "quote.create" in cap.last_system
    assert "===HOOK===" in cap.last_system


def test_chat_service_omits_quote_hook_without_active_integrations(test_settings, test_tenant) -> None:
    from datetime import UTC, datetime

    tenant, _token = test_tenant
    tenant = replace(
        tenant,
        config=TenantConfig(automation_modules=("erpnext.quote",), rag_enabled=False),
        updated_at=datetime.now(UTC),
    )
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    class CaptureLlm:
        def __init__(self) -> None:
            self.last_system: str | None = None

        def generate_chat(
            self,
            *,
            system_instruction: str,
            messages: list[ChatMessage],
            attachments: list[Attachment] | None = None,
        ) -> LlmResult:
            _ = messages, attachments
            self.last_system = system_instruction
            return LlmResult(text="ok", usage=LlmUsage())

    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(
            settings=test_settings,
            tenant=tenant,
            llm=cap,
            repo=repo,
            rag=None,
        )
        svc.handle_user_message("email:client@example.com", "Quote please")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "quote.create" not in cap.last_system
    assert "===HOOK===" in cap.last_system


def test_chat_service_injects_erp_context(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    class CaptureLlm:
        def __init__(self) -> None:
            self.last_system: str | None = None

        def generate_chat(
            self,
            *,
            system_instruction: str,
            messages: list[ChatMessage],
            attachments: list[Attachment] | None = None,
        ) -> LlmResult:
            _ = attachments
            self.last_system = system_instruction
            return LlmResult(text="ok", usage=LlmUsage())

    def enrich(session_id: str) -> str | None:
        if session_id == "whatsapp:33600000000":
            return "Customer: Test Corp\nRecent sales orders:\n- SO-1"
        return None

    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        svc = ChatService(
            settings=test_settings,
            tenant=tenant,
            llm=cap,
            repo=repo,
            rag=None,
            integration_enricher=enrich,
        )
        svc.handle_user_message("whatsapp:33600000000", "Where is my order?")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "Customer data" in cap.last_system
    assert "Test Corp" in cap.last_system


def test_chat_service_strips_marker_and_persists_hook(test_settings, test_tenant) -> None:
    from chatbot.domain.models.hook import HookStatus

    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    llm = FakeLlm(
        'Thanks, confirmed.\n===JF030A===\n{"action":"create","name":"Ana","tel":"23057770000","products":[{"qty":2,"product":"Diffuser"}]}'
    )
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        hooks = SqlAlchemyHookEventRepository(session, tenant.id)
        svc = ChatService(
            settings=test_settings,
            tenant=tenant,
            llm=llm,
            repo=repo,
            rag=None,
            hook_repo=hooks,
        )
        out = svc.handle_user_message("s-order", "I want 2 diffusers")
        session.commit()
        msgs = repo.list_messages("s-order", limit=10)
        pending = hooks.list_by_tenant(status=HookStatus.PENDING)
    finally:
        session.close()

    assert out.text == "Thanks, confirmed."
    assert msgs[-1].content == "Thanks, confirmed."
    assert len(pending) == 1
    assert pending[0].type == "order"
