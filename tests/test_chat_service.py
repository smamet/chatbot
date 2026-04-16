from __future__ import annotations

import uuid
from pathlib import Path

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.application.chat_service import ChatService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.domain.contracts.llm_client import LlmResult, LlmUsage
from chatbot.domain.models.message import ChatMessage, MessageRole


class FakeLlm:
    def __init__(self, reply: str = "hello") -> None:
        self._reply = reply

    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
    ) -> LlmResult:
        _ = system_instruction
        _ = messages
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
        ) -> LlmResult:
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
