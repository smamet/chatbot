from __future__ import annotations

import logging

from evenor.application.usage_recorder_service import UsageRecorderService
from evenor.domain.contracts.llm_client import LlmClient, LlmResult
from evenor.domain.models.api_usage import ApiUsageOperation
from evenor.domain.models.attachment import Attachment
from evenor.domain.models.message import ChatMessage

logger = logging.getLogger(__name__)


class MeteredLlmClient:
    def __init__(
        self,
        *,
        inner: LlmClient,
        tenant_id: int,
        operation: ApiUsageOperation,
        model: str,
        recorder: UsageRecorderService,
    ) -> None:
        self._inner = inner
        self._tenant_id = tenant_id
        self._operation = operation
        self._model = model
        self._recorder = recorder

    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
        attachments: list[Attachment] | None = None,
    ) -> LlmResult:
        result = self._inner.generate_chat(
            system_instruction=system_instruction,
            messages=messages,
            attachments=attachments,
        )
        self._recorder.record(
            self._tenant_id,
            self._operation,
            self._model,
            result.usage,
        )
        return result
