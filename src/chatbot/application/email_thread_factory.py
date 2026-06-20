from __future__ import annotations

from sqlalchemy.orm import Session

from chatbot.adapters.llm.gemini_client import GeminiLlmClient
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.email_thread_disambiguator import EmailThreadDisambiguator
from chatbot.application.email_thread_resolver import EmailThreadResolver
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.application.usage_metering import metered_llm
from chatbot.config.settings import Settings
from chatbot.domain.models.tenant import Tenant


def _gemini_api_key(tenant: Tenant, settings: Settings) -> str | None:
    key = (tenant.gemini_api_key or settings.gemini_api_key or "").strip()
    return key or None


def build_email_thread_resolver(
    session: Session,
    settings: Settings,
    tenant: Tenant,
) -> EmailThreadResolver:
    merged = merge_tenant_settings(settings, tenant)
    api_key = _gemini_api_key(tenant, settings)
    disambiguator: EmailThreadDisambiguator | None = None
    if merged.email_thread_llm_enabled and api_key:
        llm = metered_llm(
            inner=GeminiLlmClient(model=merged.rewrite_model, api_key=api_key),
            tenant_id=tenant.id,
            operation="email_thread",
            model=merged.rewrite_model,
            session=session,
        )
        disambiguator = EmailThreadDisambiguator(
            llm=llm,
            min_confidence=merged.email_thread_llm_min_confidence,
            enabled=True,
        )
    else:
        disambiguator = EmailThreadDisambiguator(
            llm=None,
            min_confidence=merged.email_thread_llm_min_confidence,
            enabled=False,
        )
    return EmailThreadResolver(
        session,
        tenant_id=tenant.id,
        settings=merged,
        disambiguator=disambiguator,
    )
