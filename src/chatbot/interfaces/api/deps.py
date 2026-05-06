from __future__ import annotations

import hmac
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.order_repository import SqlAlchemyOrderRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.adapters.system_clock import SystemClock
from chatbot.application.admin_notifier import NullAdminNotifier, WhatsAppAdminNotifier
from chatbot.application.chat_service import ChatService
from chatbot.application.order_service import OrderService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.config.settings import Settings, get_settings


def get_settings_dep() -> Settings:
    return get_settings()


def require_chat_api_auth(
    settings: Settings = Depends(get_settings_dep),
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """When CHAT_API_SECRET is set, require ``Authorization: Bearer <secret>`` for /v1/chat."""
    secret = settings.chat_api_secret.strip()
    if not secret:
        return
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.removeprefix("Bearer ").strip()
    a, b = token.encode("utf-8"), secret.encode("utf-8")
    if len(a) != len(b) or not hmac.compare_digest(a, b):
        raise HTTPException(status_code=401, detail="Unauthorized")


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


def get_order_repo(session: Session = Depends(get_session)) -> SqlAlchemyOrderRepository:
    return SqlAlchemyOrderRepository(session)


def get_admin_notifier(settings: Settings = Depends(get_settings_dep)):
    if (
        settings.whatsapp_phone_number_id.strip()
        and settings.whatsapp_access_token.strip()
        and settings.whatsapp_admin_wa_id.strip()
    ):
        return WhatsAppAdminNotifier(
            phone_number_id=settings.whatsapp_phone_number_id,
            access_token=settings.whatsapp_access_token,
            admin_wa_id=settings.whatsapp_admin_wa_id,
        )
    return NullAdminNotifier()


def get_order_service(
    settings: Settings = Depends(get_settings_dep),
    repo: SqlAlchemyOrderRepository = Depends(get_order_repo),
    notifier=Depends(get_admin_notifier),
) -> OrderService:
    return OrderService(
        repository=repo,
        notifier=notifier,
        clock=SystemClock(),
        modification_window_hours=settings.order_modification_window_hours,
    )


def get_chat_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    repo: SqlAlchemyConversationRepository = Depends(get_conversation_repo),
    order_service: OrderService = Depends(get_order_service),
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
        order_service=order_service,
        prompt_path=settings.prompt_path,
    )
