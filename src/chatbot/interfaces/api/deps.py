from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.chat_service import ChatService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.config.settings import Settings, get_settings


def get_settings_dep() -> Settings:
    return get_settings()


def get_session(request: Request) -> Generator[Session, None, None]:
    factory = request.app.state.session_factory
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_conversation_repo(session: Session = Depends(get_session)) -> SqlAlchemyConversationRepository:
    return SqlAlchemyConversationRepository(session)


def get_chat_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    repo: SqlAlchemyConversationRepository = Depends(get_conversation_repo),
) -> ChatService:
    rag: RagPipeline | None = None
    if settings.rag_enabled:
        rag = RagPipeline(
            settings=settings,
            rewriter_llm=request.app.state.rewriter_llm,
            embedder=request.app.state.embedder,
            vector_store=request.app.state.vector_store,
            rewrite_language_gate=getattr(request.app.state, "rewrite_language_gate", None),
        )
    return ChatService(
        settings=settings,
        llm=request.app.state.llm,
        repo=repo,
        rag=rag,
        prompt_path=settings.prompt_path,
    )
