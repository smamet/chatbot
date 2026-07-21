from __future__ import annotations

from chatbot.application.usage_recorder_service import UsageRecorderService
from chatbot.domain.contracts.embedder import Embedder
from chatbot.domain.contracts.llm_client import LlmUsage
from chatbot.domain.models.api_usage import ApiUsageOperation


class MeteredEmbedder:
    def __init__(
        self,
        *,
        inner: Embedder,
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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._inner.embed_texts(texts)
        if not texts:
            return vectors
        model = self._model
        usage = LlmUsage()
        meta = getattr(self._inner, "last_embed_meta", None)
        if meta is not None:
            model, usage = meta
        self._recorder.record(self._tenant_id, self._operation, model, usage)
        return vectors
