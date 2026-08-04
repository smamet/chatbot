from __future__ import annotations

from sqlalchemy.orm import Session

from evenor.adapters.embeddings.metered_embedder import MeteredEmbedder
from evenor.adapters.llm.metered_llm_client import MeteredLlmClient
from evenor.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository
from evenor.application.usage_recorder_service import UsageRecorderService
from evenor.domain.contracts.embedder import Embedder
from evenor.domain.contracts.llm_client import LlmClient
from evenor.domain.models.api_usage import ApiUsageOperation


def usage_recorder_for(session: Session | None) -> UsageRecorderService | None:
    if session is None:
        return None
    return UsageRecorderService(SqlAlchemyApiUsageRepository(session))


def metered_llm(
    *,
    inner: LlmClient,
    tenant_id: int,
    operation: ApiUsageOperation,
    model: str,
    session: Session | None,
) -> LlmClient:
    recorder = usage_recorder_for(session)
    if recorder is None:
        return inner
    return MeteredLlmClient(
        inner=inner,
        tenant_id=tenant_id,
        operation=operation,
        model=model,
        recorder=recorder,
    )


def metered_embedder(
    *,
    inner: Embedder,
    tenant_id: int,
    operation: ApiUsageOperation,
    model: str,
    session: Session | None,
) -> Embedder:
    recorder = usage_recorder_for(session)
    if recorder is None:
        return inner
    return MeteredEmbedder(
        inner=inner,
        tenant_id=tenant_id,
        operation=operation,
        model=model,
        recorder=recorder,
    )
