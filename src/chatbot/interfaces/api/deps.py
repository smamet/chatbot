from __future__ import annotations

import hmac
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.adapters.persistence.tenant_paths import tenant_lancedb_dir
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.chat_service import ChatService
from chatbot.application.connector_service import ConnectorService
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.application.integration_enrichment import build_enricher
from chatbot.application.integration_service import IntegrationService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.application.tenant_service import TenantService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.application.usage_metering import metered_embedder, metered_llm
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.models.tenant import Tenant


def get_settings_dep() -> Settings:
    return get_settings()


def require_admin_auth(
    settings: Settings = Depends(get_settings_dep),
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    secret = settings.admin_token.strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Admin API not configured")
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


def get_tenant_service(session: Session = Depends(get_session)) -> TenantService:
    return TenantService(SqlAlchemyTenantRepository(session))


def get_connector_service(session: Session = Depends(get_session)) -> ConnectorService:
    return ConnectorService(SqlAlchemyConnectorRepository(session))


def get_tenant_by_slug(
    slug: str,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> Tenant:
    tenant = tenant_service.get_by_slug(slug)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if not tenant.active:
        raise HTTPException(status_code=403, detail="Tenant inactive")
    return tenant


def get_current_tenant(
    slug: str,
    authorization: Annotated[str | None, Header()] = None,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> Tenant:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.removeprefix("Bearer ").strip()
    tenant = tenant_service.get_by_token(token)
    if tenant is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if tenant.slug != slug:
        raise HTTPException(status_code=403, detail="Token does not match tenant")
    if not tenant.active:
        raise HTTPException(status_code=403, detail="Tenant inactive")
    return tenant


def _gemini_api_key(tenant: Tenant, settings: Settings) -> str:
    return (tenant.gemini_api_key or settings.gemini_api_key or "").strip()


def _llm_for_tenant(request: Request, tenant: Tenant, settings: Settings) -> GeminiLlmClient:
    key = _gemini_api_key(tenant, settings)
    merged = merge_tenant_settings(settings, tenant)
    cache: dict[str, GeminiLlmClient] = request.app.state.llm_by_key
    if key not in cache:
        cache[key] = GeminiLlmClient(model=merged.chat_model, api_key=key or None)
    return cache[key]


def _embedder_for_tenant(request: Request, tenant: Tenant, settings: Settings) -> GeminiEmbedder:
    key = _gemini_api_key(tenant, settings)
    merged = merge_tenant_settings(settings, tenant)
    cache: dict[str, GeminiEmbedder] = request.app.state.embedder_by_key
    if key not in cache:
        cache[key] = GeminiEmbedder(api_key=key or None, model=merged.embedding_model)
    return cache[key]


def _vector_store_for_tenant(request: Request, tenant: Tenant, settings: Settings) -> LanceVectorStore:
    stores: dict[str, LanceVectorStore] = request.app.state.vector_stores
    if tenant.slug not in stores:
        stores[tenant.slug] = LanceVectorStore(tenant_lancedb_dir(settings, tenant.slug))
    return stores[tenant.slug]


def _build_chat_service(
    request: Request,
    settings: Settings,
    tenant: Tenant,
    repo: SqlAlchemyConversationRepository,
    hook_repo: SqlAlchemyHookEventRepository,
    *,
    db_session: Session | None = None,
) -> ChatService:
    merged = merge_tenant_settings(settings, tenant)
    rag: RagPipeline | None = None
    api_key = _gemini_api_key(tenant, settings) or None
    if merged.rag_enabled:
        rag = RagPipeline(
            settings=merged,
            rewriter_llm=metered_llm(
                inner=GeminiLlmClient(model=merged.rewrite_model, api_key=api_key),
                tenant_id=tenant.id,
                operation="rewrite",
                model=merged.rewrite_model,
                session=db_session,
            ),
            embedder=metered_embedder(
                inner=_embedder_for_tenant(request, tenant, settings),
                tenant_id=tenant.id,
                operation="embed_chat",
                model=merged.embedding_model,
                session=db_session,
            ),
            vector_store=_vector_store_for_tenant(request, tenant, settings),
            rewrite_language_gate=getattr(request.app.state, "rewrite_language_gate", None),
        )
    integration_enricher = build_enricher(db_session, tenant.id) if db_session else None
    active_integrations = (
        IntegrationService(SqlAlchemyIntegrationRepository(db_session)).active_types_for_tenant(
            tenant.id
        )
        if db_session
        else None
    )
    return ChatService(
        settings=settings,
        tenant=tenant,
        llm=metered_llm(
            inner=_llm_for_tenant(request, tenant, settings),
            tenant_id=tenant.id,
            operation="chat",
            model=merged.chat_model,
            session=db_session,
        ),
        repo=repo,
        rag=rag,
        hook_repo=hook_repo,
        integration_enricher=integration_enricher,
        active_integrations=active_integrations,
    )


def get_conversation_repo(
    session: Session = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
) -> SqlAlchemyConversationRepository:
    return SqlAlchemyConversationRepository(session, tenant.id)


def get_hook_repo(
    session: Session = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
) -> SqlAlchemyHookEventRepository:
    return SqlAlchemyHookEventRepository(session, tenant.id)


def get_chat_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    tenant: Tenant = Depends(get_current_tenant),
    repo: SqlAlchemyConversationRepository = Depends(get_conversation_repo),
    hook_repo: SqlAlchemyHookEventRepository = Depends(get_hook_repo),
    session: Session = Depends(get_session),
) -> ChatService:
    return _build_chat_service(
        request, settings, tenant, repo, hook_repo, db_session=session
    )


def get_webhook_tenant(
    slug: str,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> Tenant:
    return get_tenant_by_slug(slug, tenant_service)


def get_webhook_conversation_repo(
    session: Session = Depends(get_session),
    tenant: Tenant = Depends(get_webhook_tenant),
) -> SqlAlchemyConversationRepository:
    return SqlAlchemyConversationRepository(session, tenant.id)


def get_webhook_hook_repo(
    session: Session = Depends(get_session),
    tenant: Tenant = Depends(get_webhook_tenant),
) -> SqlAlchemyHookEventRepository:
    return SqlAlchemyHookEventRepository(session, tenant.id)


def get_webhook_chat_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    tenant: Tenant = Depends(get_webhook_tenant),
    repo: SqlAlchemyConversationRepository = Depends(get_webhook_conversation_repo),
    hook_repo: SqlAlchemyHookEventRepository = Depends(get_webhook_hook_repo),
    session: Session = Depends(get_session),
) -> ChatService:
    return _build_chat_service(
        request, settings, tenant, repo, hook_repo, db_session=session
    )
