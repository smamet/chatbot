from __future__ import annotations

import logging
from datetime import UTC, datetime

from chatbot.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository
from chatbot.domain.contracts.llm_client import LlmUsage
from chatbot.domain.models.api_usage import ApiUsageDayEntry, ApiUsageOperation, ApiUsageSummary

logger = logging.getLogger(__name__)


class UsageRecorderService:
    def __init__(self, repo: SqlAlchemyApiUsageRepository) -> None:
        self._repo = repo

    def record(
        self,
        tenant_id: int,
        operation: ApiUsageOperation,
        model: str,
        usage: LlmUsage,
        *,
        call_count: int = 1,
    ) -> None:
        try:
            prompt = usage.prompt_tokens or 0
            output = usage.candidates_tokens or 0
            total = usage.total_tokens if usage.total_tokens is not None else prompt + output
            self._repo.increment(
                tenant_id=tenant_id,
                usage_date=datetime.now(UTC).date(),
                operation=operation,
                model=model or "",
                prompt_tokens=prompt,
                output_tokens=output,
                total_tokens=total,
                call_count=call_count,
            )
        except Exception:
            logger.exception(
                "Failed to record API usage tenant_id=%s operation=%s model=%s",
                tenant_id,
                operation,
                model,
            )

    def tenant_summary_since(self, tenant_id: int, since) -> ApiUsageSummary:
        return self._repo.tenant_summary_since(tenant_id, since)

    def tenant_daily_since(self, tenant_id: int, since) -> list[ApiUsageDayEntry]:
        return self._repo.tenant_daily_since(tenant_id, since)

    def all_tenant_summaries_since(self, since) -> dict[int, ApiUsageSummary]:
        return self._repo.all_tenant_summaries_since(since)

    def tenant_token_series_since(self, tenant_id: int, since):
        return self._repo.tenant_token_series_since(tenant_id, since)

    def platform_token_series_since(self, since):
        return self._repo.platform_token_series_since(since)
