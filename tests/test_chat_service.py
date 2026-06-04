from __future__ import annotations

import uuid
from pathlib import Path

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.application.order_service import OrderServiceResult
from chatbot.application.chat_service import ChatService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.domain.contracts.llm_client import LlmResult, LlmUsage
from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.domain.models.order import OrderAction, OrderCommand


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


def test_chat_service_roundtrip(test_settings, tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are a test bot.", encoding="utf-8")
    test_settings = test_settings.model_copy(update={"prompt_path": prompt_file})
    engine = create_db_engine(test_settings)
    factory = session_factory(engine)
    fake = FakeLlm("Thanks for your message.")
    sid = f"sess-{uuid.uuid4().hex}"
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session)
        svc = ChatService(settings=test_settings, llm=fake, repo=repo, rag=None, prompt_path=prompt_file)
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
    test_settings, tmp_path: Path
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are a test bot.", encoding="utf-8")
    settings = test_settings.model_copy(update={"prompt_path": prompt_file})
    engine = create_db_engine(settings)
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
        repo = SqlAlchemyConversationRepository(session)
        svc = ChatService(settings=settings, llm=cap, repo=repo, rag=None, prompt_path=prompt_file)
        svc.handle_user_message(sid, "See attached", attachments=[att])
        session.commit()
        msgs = repo.list_messages(sid, limit=10)
    finally:
        session.close()

    assert msgs[0].content == "See attached\n[Attached: quote.pdf]"
    assert cap.last_attachments is not None
    assert len(cap.last_attachments) == 1
    assert cap.last_attachments[0].filename == "quote.pdf"


def test_chat_service_includes_rag_context(test_settings, tmp_path: Path) -> None:
    from chatbot.domain.contracts.vector_store import RetrievedChunk, VectorRecord

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

    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are support.", encoding="utf-8")
    settings = test_settings.model_copy(
        update={"prompt_path": prompt_file, "rag_enabled": True, "rag_rewrite_enabled": False}
    )
    engine = create_db_engine(settings)
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
        settings=settings,
        rewriter_llm=rewriter,
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
    )
    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session)
        svc = ChatService(settings=settings, llm=cap, repo=repo, rag=rag, prompt_path=prompt_file)
        svc.handle_user_message("s2", "How much?")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "Retrieved context" in cap.last_system
    assert "42 EUR" in cap.last_system
    assert "/tmp/price.csv" not in cap.last_system
    assert "Do not mention internal file names" in cap.last_system


def test_chat_service_rag_includes_source_paths_when_dev_mode(test_settings, tmp_path: Path) -> None:
    from chatbot.domain.contracts.vector_store import RetrievedChunk, VectorRecord

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

    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are support.", encoding="utf-8")
    settings = test_settings.model_copy(
        update={
            "prompt_path": prompt_file,
            "rag_enabled": True,
            "rag_rewrite_enabled": False,
            "dev_mode": True,
        }
    )
    engine = create_db_engine(settings)
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
        settings=settings,
        rewriter_llm=rewriter,
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
    )
    cap = CaptureLlm()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session)
        svc = ChatService(settings=settings, llm=cap, repo=repo, rag=rag, prompt_path=prompt_file)
        svc.handle_user_message("s3", "How much?")
        session.commit()
    finally:
        session.close()
    assert cap.last_system is not None
    assert "/tmp/price.csv" in cap.last_system
    assert "chunk c1" in cap.last_system
    assert "Do not mention internal file names" not in cap.last_system


def test_chat_service_strips_marker_and_calls_order_service(test_settings, tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are support.", encoding="utf-8")
    settings = test_settings.model_copy(update={"prompt_path": prompt_file})
    engine = create_db_engine(settings)
    factory = session_factory(engine)

    class FakeOrderService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def append_command(self, **kwargs):
            self.calls.append(kwargs)
            return OrderServiceResult(
                action=OrderAction.CREATE,
                result="created",
                order=None,
                message="ok",
            )

    llm = FakeLlm(
        'Thanks, confirmed.\n===JF030A===\n{"action":"create","name":"Ana","tel":"23057770000","products":[{"qty":2,"product":"Diffuser"}]}'
    )
    fake_order_service = FakeOrderService()
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session)
        svc = ChatService(
            settings=settings,
            llm=llm,
            repo=repo,
            rag=None,
            order_service=fake_order_service,  # type: ignore[arg-type]
            prompt_path=prompt_file,
        )
        out = svc.handle_user_message("s-order", "I want 2 diffusers")
        session.commit()
        msgs = repo.list_messages("s-order", limit=10)
    finally:
        session.close()

    assert out.text == "Thanks, confirmed."
    assert msgs[-1].content == "Thanks, confirmed."
    assert len(fake_order_service.calls) == 1
    call = fake_order_service.calls[0]
    assert call["session_id"] == "s-order"
    assert isinstance(call["command"], OrderCommand)
    assert len(call["conversation_context"]) == 2


def test_chat_service_passes_only_last_six_messages_to_order_service(test_settings, tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are support.", encoding="utf-8")
    settings = test_settings.model_copy(update={"prompt_path": prompt_file})
    engine = create_db_engine(settings)
    factory = session_factory(engine)

    class FakeOrderService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def append_command(self, **kwargs):
            self.calls.append(kwargs)
            return OrderServiceResult(action=OrderAction.CREATE, result="created", order=None, message="ok")

    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session)
        for i in range(8):
            repo.append_message("s6", ChatMessage(role=MessageRole.USER, content=f"old-{i}"))
        llm = FakeLlm(
            'Sure.\n===JF030A===\n{"action":"create","tel":"23057770000","products":[{"qty":1,"product":"Mist"}]}'
        )
        fake_order_service = FakeOrderService()
        svc = ChatService(
            settings=settings,
            llm=llm,
            repo=repo,
            rag=None,
            order_service=fake_order_service,  # type: ignore[arg-type]
            prompt_path=prompt_file,
        )
        svc.handle_user_message("s6", "new-order")
        session.commit()
    finally:
        session.close()

    context = fake_order_service.calls[0]["conversation_context"]
    assert len(context) == 6
