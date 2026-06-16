from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.sync_service import IngestSyncService
from chatbot.application.tenant_service import TenantService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.application.usage_metering import metered_embedder
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.models.tenant import Tenant, TenantConfig
from chatbot.interfaces.api.deps import get_session, get_settings_dep, get_tenant_service, require_admin_auth

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_auth)])


class TenantConfigOut(BaseModel):
    chat_model: str
    embedding_model: str
    rewrite_model: str
    rag_enabled: bool
    rag_rewrite_enabled: bool
    rag_rewrite_lang_filter: bool
    rag_top_k: int
    chunk_size: int
    chunk_overlap: int
    retrieval_language: str
    dev_mode: bool


class TenantOut(BaseModel):
    id: int
    slug: str
    name: str
    prompt: str
    config: TenantConfigOut
    active: bool


class TenantCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    slug: str | None = Field(default=None, max_length=64)
    prompt: str = "You are a helpful assistant."
    config: TenantConfigOut | None = None


class TenantCreateOut(BaseModel):
    tenant: TenantOut
    token: str


class TenantUpdateIn(BaseModel):
    name: str | None = None
    prompt: str | None = None
    config: TenantConfigOut | None = None
    active: bool | None = None


class TokenRegenerateOut(BaseModel):
    tenant: TenantOut
    token: str


class SyncOut(BaseModel):
    logs: list[str]


class SessionOut(BaseModel):
    session_id: str


class MessageOut(BaseModel):
    role: str
    content: str


def _config_out(cfg: TenantConfig) -> TenantConfigOut:
    return TenantConfigOut(
        chat_model=cfg.chat_model,
        embedding_model=cfg.embedding_model,
        rewrite_model=cfg.rewrite_model,
        rag_enabled=cfg.rag_enabled,
        rag_rewrite_enabled=cfg.rag_rewrite_enabled,
        rag_rewrite_lang_filter=cfg.rag_rewrite_lang_filter,
        rag_top_k=cfg.rag_top_k,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        retrieval_language=cfg.retrieval_language,
        dev_mode=cfg.dev_mode,
    )


def _tenant_out(t: Tenant) -> TenantOut:
    return TenantOut(
        id=t.id,
        slug=t.slug,
        name=t.name,
        prompt=t.prompt,
        config=_config_out(t.config),
        active=t.active,
    )


def _config_in_to_domain(cfg: TenantConfigOut | None) -> TenantConfig | None:
    if cfg is None:
        return None
    return TenantConfig(
        chat_model=cfg.chat_model,
        embedding_model=cfg.embedding_model,
        rewrite_model=cfg.rewrite_model,
        rag_enabled=cfg.rag_enabled,
        rag_rewrite_enabled=cfg.rag_rewrite_enabled,
        rag_rewrite_lang_filter=cfg.rag_rewrite_lang_filter,
        rag_top_k=cfg.rag_top_k,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        retrieval_language=cfg.retrieval_language,
        dev_mode=cfg.dev_mode,
    )


@router.get("/tenants", response_model=list[TenantOut])
def list_tenants(svc: TenantService = Depends(get_tenant_service)) -> list[TenantOut]:
    return [_tenant_out(t) for t in svc.list_tenants()]


@router.post("/tenants", response_model=TenantCreateOut)
def create_tenant(
    body: TenantCreateIn,
    svc: TenantService = Depends(get_tenant_service),
) -> TenantCreateOut:
    result = svc.create_tenant(
        name=body.name,
        slug=body.slug,
        prompt=body.prompt,
        config=_config_in_to_domain(body.config),
    )
    return TenantCreateOut(tenant=_tenant_out(result.tenant), token=result.token)


@router.get("/tenants/{tenant_id}", response_model=TenantOut)
def get_tenant(tenant_id: int, svc: TenantService = Depends(get_tenant_service)) -> TenantOut:
    tenant = svc.get_by_id(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return _tenant_out(tenant)


@router.patch("/tenants/{tenant_id}", response_model=TenantOut)
def update_tenant(
    tenant_id: int,
    body: TenantUpdateIn,
    svc: TenantService = Depends(get_tenant_service),
) -> TenantOut:
    tenant = svc.update_tenant(
        tenant_id,
        name=body.name,
        prompt=body.prompt,
        config=_config_in_to_domain(body.config),
        active=body.active,
    )
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return _tenant_out(tenant)


@router.post("/tenants/{tenant_id}/regenerate-token", response_model=TokenRegenerateOut)
def regenerate_token(
    tenant_id: int,
    svc: TenantService = Depends(get_tenant_service),
) -> TokenRegenerateOut:
    out = svc.regenerate_token(tenant_id)
    if out is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    tenant, token = out
    return TokenRegenerateOut(tenant=_tenant_out(tenant), token=token)


def _tenant_by_slug(svc: TenantService, slug: str) -> Tenant:
    tenant = svc.get_by_slug(slug)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return tenant


@router.get("/tenants/slug/{slug}/documents")
def list_documents(
    slug: str,
    settings: Settings = Depends(get_settings_dep),
    svc: TenantService = Depends(get_tenant_service),
) -> list[str]:
    _tenant_by_slug(svc, slug)
    docs = tenant_docs_dir(settings, slug)
    return sorted(str(p.relative_to(docs)) for p in docs.rglob("*") if p.is_file())


@router.post("/tenants/slug/{slug}/documents")
async def upload_documents(
    slug: str,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings_dep),
    svc: TenantService = Depends(get_tenant_service),
) -> dict[str, list[str]]:
    _tenant_by_slug(svc, slug)
    docs = tenant_docs_dir(settings, slug)
    saved: list[str] = []
    for f in files:
        name = Path(f.filename or "upload.bin").name
        dest = docs / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await f.read())
        saved.append(name)
    return {"saved": saved}


@router.post("/tenants/slug/{slug}/sync", response_model=SyncOut)
def sync_tenant_docs(
    slug: str,
    fresh: bool = False,
    settings: Settings = Depends(get_settings_dep),
    svc: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
) -> SyncOut:
    tenant = _tenant_by_slug(svc, slug)
    merged = merge_tenant_settings(settings, tenant)
    docs = tenant_docs_dir(settings, slug)
    store = LanceVectorStore(settings.lancedb_root / slug)
    embedder = metered_embedder(
        inner=GeminiEmbedder(),
        tenant_id=tenant.id,
        operation="embed_ingest",
        model=merged.embedding_model,
        session=session,
    )
    sync_svc = IngestSyncService(
        settings=merged,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    logs = sync_svc.reconcile_root(docs, fresh=fresh)
    return SyncOut(logs=logs)


@router.get("/tenants/slug/{slug}/sessions", response_model=list[SessionOut])
def list_sessions(
    slug: str,
    svc: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
) -> list[SessionOut]:
    tenant = _tenant_by_slug(svc, slug)
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    return [SessionOut(session_id=s) for s in repo.list_session_ids()]


@router.get("/tenants/slug/{slug}/sessions/{session_id}/messages", response_model=list[MessageOut])
def list_session_messages(
    slug: str,
    session_id: str,
    svc: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
) -> list[MessageOut]:
    tenant = _tenant_by_slug(svc, slug)
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    msgs = repo.list_messages(session_id, limit=500)
    return [MessageOut(role=m.role.value, content=m.content) for m in msgs]
