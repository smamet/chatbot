from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


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
    automation_modules: tuple[str, ...] = ("core.orders",)
    hook_instructions_extra: str = ""

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
                "automation_modules": list(self.automation_modules),
                "hook_instructions_extra": self.hook_instructions_extra,
            },
            ensure_ascii=True,
        )

    def resolved_automation_modules(self, legacy_hook_instructions: str | None) -> list[str]:
        if self.automation_modules:
            return list(self.automation_modules)
        if legacy_hook_instructions and str(legacy_hook_instructions).strip():
            return ["core.orders"]
        return []

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
        modules_raw = data.get("automation_modules")
        if "automation_modules" in data and isinstance(modules_raw, list):
            automation_modules = tuple(str(m).strip() for m in modules_raw if str(m).strip())
        elif "automation_modules" not in data:
            automation_modules = cls().automation_modules
        else:
            automation_modules = ()
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
            automation_modules=automation_modules,
            hook_instructions_extra=str(data.get("hook_instructions_extra", "")),
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
    client_billing_input_per_million_usd: Decimal | None = None
    client_billing_output_per_million_usd: Decimal | None = None

    @property
    def hooks_enabled(self) -> bool:
        from chatbot.application.hook_prompt_composer import hooks_enabled_for_tenant

        return hooks_enabled_for_tenant(self)


@dataclass(frozen=True)
class TenantCreateResult:
    tenant: Tenant
    token: str
