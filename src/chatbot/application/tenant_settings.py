from __future__ import annotations

from chatbot.config.settings import Settings
from chatbot.domain.models.tenant import Tenant, TenantConfig


def merge_tenant_settings(global_settings: Settings, tenant: Tenant) -> Settings:
    """Overlay tenant RAG/model flags onto global settings for ChatService/RAG."""
    cfg: TenantConfig = tenant.config
    return global_settings.model_copy(
        update={
            "chat_model": cfg.chat_model,
            "embedding_model": cfg.embedding_model,
            "rewrite_model": cfg.rewrite_model,
            "rag_enabled": cfg.rag_enabled,
            "rag_rewrite_enabled": cfg.rag_rewrite_enabled,
            "rag_rewrite_lang_filter": cfg.rag_rewrite_lang_filter,
            "rag_top_k": cfg.rag_top_k,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "retrieval_language": cfg.retrieval_language,
            "dev_mode": cfg.dev_mode,
        }
    )
