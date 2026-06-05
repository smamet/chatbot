from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TenantConfig:
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    rewrite_model: str = "gemini-2.5-flash"
    rag_enabled: bool = True
    rag_rewrite_enabled: bool = True
    rag_rewrite_lang_filter: bool = True
    rag_top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_language: str = "en"
    dev_mode: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "chat_model": self.chat_model,
                "embedding_model": self.embedding_model,
                "rewrite_model": self.rewrite_model,
                "rag_enabled": self.rag_enabled,
                "rag_rewrite_enabled": self.rag_rewrite_enabled,
                "rag_rewrite_lang_filter": self.rag_rewrite_lang_filter,
                "rag_top_k": self.rag_top_k,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "retrieval_language": self.retrieval_language,
                "dev_mode": self.dev_mode,
            },
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> TenantConfig:
        if not raw or not raw.strip():
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls(
            chat_model=str(data.get("chat_model", cls.chat_model)),
            embedding_model=str(data.get("embedding_model", cls.embedding_model)),
            rewrite_model=str(data.get("rewrite_model", cls.rewrite_model)),
            rag_enabled=bool(data.get("rag_enabled", cls.rag_enabled)),
            rag_rewrite_enabled=bool(data.get("rag_rewrite_enabled", True)),
            rag_rewrite_lang_filter=bool(data.get("rag_rewrite_lang_filter", cls.rag_rewrite_lang_filter)),
            rag_top_k=int(data.get("rag_top_k", 5)),
            chunk_size=int(data.get("chunk_size", 800)),
            chunk_overlap=int(data.get("chunk_overlap", 100)),
            retrieval_language=str(data.get("retrieval_language", "en")),
            dev_mode=bool(data.get("dev_mode", False)),
        )


@dataclass(frozen=True)
class Tenant:
    id: int
    slug: str
    name: str
    prompt: str
    hook_instructions: str | None
    gemini_api_key: str | None
    config: TenantConfig
    active: bool
    created_at: datetime
    updated_at: datetime

    @property
    def hooks_enabled(self) -> bool:
        return bool((self.hook_instructions or "").strip())


@dataclass(frozen=True)
class TenantCreateResult:
    tenant: Tenant
    token: str
