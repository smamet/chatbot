from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class BotType(StrEnum):
    ASSISTANT = "assistant"
    TRADER = "trader"


@dataclass(frozen=True)
class TraderSettings:
    """First-class trading bot settings (not an Integrations row)."""

    market_profile: str = "cac40"
    symbol: str = "CAC40"
    epic: str = "IX.D.CAC.BMU.IP"
    fundmanager_url: str = ""
    fundmanager_token: str = ""
    max_open_positions: int = 4
    # Cached from IG resolve on bot create/update (display + live defaults).
    pnl_currency: str = ""
    point_value: float = 0.0  # 0 = unresolved; live derives from IG

    def to_dict(self) -> dict:
        return {
            "market_profile": self.market_profile,
            "symbol": self.symbol,
            "epic": self.epic,
            "fundmanager_url": self.fundmanager_url,
            "fundmanager_token": self.fundmanager_token,
            "max_open_positions": self.max_open_positions,
            "pnl_currency": self.pnl_currency,
            "point_value": self.point_value,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> TraderSettings:
        if not isinstance(data, dict):
            return cls()
        try:
            max_legs = int(data.get("max_open_positions", cls.max_open_positions))
        except (TypeError, ValueError):
            max_legs = cls.max_open_positions
        try:
            point_value = float(data.get("point_value") or 0.0)
        except (TypeError, ValueError):
            point_value = 0.0
        return cls(
            market_profile=str(data.get("market_profile") or cls.market_profile).strip()
            or cls.market_profile,
            symbol=str(data.get("symbol") or cls.symbol).strip() or cls.symbol,
            epic=str(data.get("epic") or cls.epic).strip() or cls.epic,
            fundmanager_url=str(data.get("fundmanager_url") or "").strip(),
            fundmanager_token=str(data.get("fundmanager_token") or ""),
            max_open_positions=max(1, max_legs),
            pnl_currency=str(data.get("pnl_currency") or "").strip().upper(),
            point_value=point_value if point_value > 0 else 0.0,
        )


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
    email_blocked_senders: tuple[str, ...] = ()
    # Empty = all connector type×direction capabilities allowed (backward compatible).
    allowed_connectors: tuple[str, ...] = ()
    # Empty = all integration types allowed (backward compatible).
    allowed_integrations: tuple[str, ...] = ()
    trader: TraderSettings = field(default_factory=TraderSettings)

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
                "email_blocked_senders": list(self.email_blocked_senders),
                "allowed_connectors": list(self.allowed_connectors),
                "allowed_integrations": list(self.allowed_integrations),
                "trader": self.trader.to_dict(),
            },
            ensure_ascii=True,
        )

    def resolved_automation_modules(self, legacy_hook_instructions: str | None) -> list[str]:
        if self.automation_modules:
            return list(self.automation_modules)
        if legacy_hook_instructions and str(legacy_hook_instructions).strip():
            return ["core.orders"]
        return []

    def with_trader(self, **kwargs) -> TenantConfig:
        return replace(self, trader=replace(self.trader, **kwargs))

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
        blocked_raw = data.get("email_blocked_senders")
        if "email_blocked_senders" in data and isinstance(blocked_raw, list):
            email_blocked_senders = tuple(
                str(addr).strip().lower() for addr in blocked_raw if str(addr).strip()
            )
        else:
            email_blocked_senders = ()
        allowed_raw = data.get("allowed_connectors")
        if "allowed_connectors" in data and isinstance(allowed_raw, list):
            allowed_connectors = tuple(
                str(item).strip().lower() for item in allowed_raw if str(item).strip()
            )
        else:
            allowed_connectors = ()
        allowed_integrations_raw = data.get("allowed_integrations")
        if "allowed_integrations" in data and isinstance(allowed_integrations_raw, list):
            allowed_integrations = tuple(
                str(item).strip().lower()
                for item in allowed_integrations_raw
                if str(item).strip()
            )
        else:
            allowed_integrations = ()
        trader_raw = data.get("trader")
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
            email_blocked_senders=email_blocked_senders,
            allowed_connectors=allowed_connectors,
            allowed_integrations=allowed_integrations,
            trader=TraderSettings.from_dict(trader_raw if isinstance(trader_raw, dict) else None),
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
    bot_type: BotType = BotType.ASSISTANT
    client_billing_input_per_million_usd: Decimal | None = None
    client_billing_output_per_million_usd: Decimal | None = None

    @property
    def is_trader(self) -> bool:
        return self.bot_type == BotType.TRADER

    @property
    def is_assistant(self) -> bool:
        return self.bot_type == BotType.ASSISTANT

    @property
    def hooks_enabled(self) -> bool:
        from chatbot.application.hook_prompt_composer import hooks_enabled_for_tenant

        return hooks_enabled_for_tenant(self)


@dataclass(frozen=True)
class TenantCreateResult:
    tenant: Tenant
    token: str
