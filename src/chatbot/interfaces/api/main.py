from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.config.settings import get_settings
from chatbot.interfaces.api.error_handlers import register_exception_handlers
from chatbot.interfaces.api.routers import chat, whatsapp_webhook


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = create_db_engine(settings)
    app.state.session_factory = session_factory(engine)
    app.state.llm = GeminiLlmClient(api_key=settings.gemini_api_key, model=settings.chat_model)
    app.state.rewriter_llm = GeminiLlmClient(api_key=settings.gemini_api_key, model=settings.rewrite_model)
    app.state.embedder = GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.embedding_model)
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    app.state.vector_store = LanceVectorStore(settings.lancedb_path)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Customer chatbot", lifespan=lifespan)
    register_exception_handlers(app)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    app.include_router(whatsapp_webhook.router, tags=["whatsapp"])
    return app


app = create_app()
