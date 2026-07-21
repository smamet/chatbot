from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.rag.fasttext_language_gate import load_rewrite_language_gate
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.config.settings import Settings, get_settings
from chatbot.interfaces.api.error_handlers import register_exception_handlers
from chatbot.interfaces.api.routers import (
    admin,
    auth_web,
    cac40_web,
    chat,
    dashboard_web,
    instagram_webhook,
    messenger_webhook,
    whatsapp_webhook,
)

_STATIC = Path(__file__).resolve().parent.parent / "web" / "static"


def _configure_rag_verbose_logging(enabled: bool) -> None:
    log = logging.getLogger("chatbot")
    marker = "_rag_verbose_stream_handler"
    log.handlers = [h for h in log.handlers if not getattr(h, marker, False)]
    if not enabled:
        log.setLevel(logging.NOTSET)
        return
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    setattr(handler, marker, True)
    handler.setFormatter(logging.Formatter("[RAG-verbose] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)


def _genai_service_signature(s: Settings) -> tuple[object, ...]:
    return (
        s.chat_model,
        s.rewrite_model,
        s.embedding_model,
        s.gemini_api_key,
        s.rag_rewrite_lang_filter,
        s.rag_verbose,
        str(s.lancedb_root),
    )


def refresh_genai_clients_if_needed(app: FastAPI) -> None:
    s = get_settings()
    sig = _genai_service_signature(s)
    if getattr(app.state, "_genai_service_sig", None) == sig:
        return
    app.state.llm = GeminiLlmClient(model_attr="chat_model")
    app.state.rewriter_llm = GeminiLlmClient(model_attr="rewrite_model")
    app.state.embedder = GeminiEmbedder()
    app.state.llm_by_key: dict[str, GeminiLlmClient] = {}
    app.state.embedder_by_key: dict[str, GeminiEmbedder] = {}
    s.lancedb_root.mkdir(parents=True, exist_ok=True)
    s.data_root.mkdir(parents=True, exist_ok=True)
    app.state.vector_stores: dict[str, LanceVectorStore] = {}
    _configure_rag_verbose_logging(s.rag_verbose)
    app.state.rewrite_language_gate = load_rewrite_language_gate(s)
    app.state._genai_service_sig = sig


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = create_db_engine(settings)
    app.state.session_factory = session_factory(engine)
    refresh_genai_clients_if_needed(app)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Multi-tenant chatbot", lifespan=lifespan)
    register_exception_handlers(app)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        max_age=max(1, settings.session_max_age_days) * 24 * 60 * 60,
    )
    if _STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.middleware("http")
    async def _reload_env_clients_middleware(request: Request, call_next):
        refresh_genai_clients_if_needed(request.app)
        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth_web.router)
    app.include_router(dashboard_web.router)
    app.include_router(cac40_web.router)
    app.include_router(chat.router, tags=["chat"])
    app.include_router(admin.router, tags=["admin"])
    app.include_router(whatsapp_webhook.router, tags=["whatsapp"])
    app.include_router(messenger_webhook.router, tags=["messenger"])
    app.include_router(instagram_webhook.router, tags=["instagram"])
    return app


app = create_app()
