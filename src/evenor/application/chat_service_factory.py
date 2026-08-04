from __future__ import annotations

from sqlalchemy.orm import Session

from evenor.adapters.embeddings.gemini_embedder import GeminiEmbedder
from evenor.adapters.llm.gemini_client import GeminiLlmClient
from evenor.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from evenor.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from evenor.adapters.persistence.tenant_paths import tenant_lancedb_dir
from evenor.adapters.rag.lance_vector_store import LanceVectorStore
from evenor.application.chat_service import ChatService
from evenor.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from evenor.application.integration_enrichment import build_enricher
from evenor.application.integration_service import IntegrationService
from evenor.application.rag_orchestrator import RagPipeline
from evenor.application.tenant_settings import merge_tenant_settings
from evenor.application.usage_metering import metered_embedder, metered_llm
from evenor.config.settings import Settings
from evenor.domain.models.tenant import Tenant


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
            rewriter_llm=metered_llm(
                inner=GeminiLlmClient(model=merged.rewrite_model, api_key=api_key),
                tenant_id=tenant.id,
                operation="rewrite",
                model=merged.rewrite_model,
                session=session,
            ),
            embedder=metered_embedder(
                inner=GeminiEmbedder(api_key=api_key, model=merged.embedding_model),
                tenant_id=tenant.id,
                operation="embed_chat",
                model=merged.embedding_model,
                session=session,
            ),
            vector_store=LanceVectorStore(tenant_lancedb_dir(settings, tenant.slug)),
            rewrite_language_gate=None,
        )
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    hook_repo = SqlAlchemyHookEventRepository(session, tenant.id)
    integ_svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    return ChatService(
        settings=settings,
        tenant=tenant,
        llm=metered_llm(
            inner=GeminiLlmClient(model=merged.chat_model, api_key=api_key),
            tenant_id=tenant.id,
            operation="chat",
            model=merged.chat_model,
            session=session,
        ),
        repo=repo,
        rag=rag,
        hook_repo=hook_repo,
        integration_enricher=build_enricher(session, tenant.id),
        active_integrations=integ_svc.active_types_for_tenant(tenant.id),
    )
