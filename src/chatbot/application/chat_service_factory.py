from __future__ import annotations

from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.adapters.persistence.tenant_paths import tenant_lancedb_dir
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.chat_service import ChatService
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.application.integration_enrichment import build_enricher
from chatbot.application.integration_service import IntegrationService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.config.settings import Settings
from chatbot.domain.models.tenant import Tenant


def _gemini_api_key(tenant: Tenant, settings: Settings) -> str:
    return (tenant.gemini_api_key or settings.gemini_api_key or "").strip()


def build_chat_service_for_worker(
    session: Session,
    settings: Settings,
    tenant: Tenant,
) -> ChatService:
    merged = merge_tenant_settings(settings, tenant)
    api_key = _gemini_api_key(tenant, settings) or None
    rag: RagPipeline | None = None
    if merged.rag_enabled:
        rag = RagPipeline(
            settings=merged,
            rewriter_llm=GeminiLlmClient(model=merged.rewrite_model, api_key=api_key),
            embedder=GeminiEmbedder(api_key=api_key, model=merged.embedding_model),
            vector_store=LanceVectorStore(tenant_lancedb_dir(settings, tenant.slug)),
            rewrite_language_gate=None,
        )
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    hook_repo = SqlAlchemyHookEventRepository(session, tenant.id)
    integ_svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    return ChatService(
        settings=settings,
        tenant=tenant,
        llm=GeminiLlmClient(model=merged.chat_model, api_key=api_key),
        repo=repo,
        rag=rag,
        hook_repo=hook_repo,
        integration_enricher=build_enricher(session, tenant.id),
        active_integrations=integ_svc.active_types_for_tenant(tenant.id),
    )
