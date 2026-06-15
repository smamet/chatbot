from __future__ import annotations

import time

from google import genai
from google.genai import errors, types

from chatbot.config.settings import Settings, get_settings
from chatbot.domain.contracts.embedder import Embedder

_RETRYABLE_CODES = frozenset({429, 503})
_MAX_BACKOFF_429_SECONDS = 120.0
_MAX_BACKOFF_503_SECONDS = 30.0


def _retry_after_seconds(exc: errors.APIError) -> float | None:
    response = exc.response
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _backoff_seconds(code: int, attempt: int, settings: Settings, exc: errors.APIError) -> float:
    if code == 429:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, _MAX_BACKOFF_429_SECONDS)
        return min(settings.embed_retry_base_429_seconds * (2**attempt), _MAX_BACKOFF_429_SECONDS)
    if code == 503:
        return min(settings.embed_retry_base_503_seconds * (2**attempt), _MAX_BACKOFF_503_SECONDS)
    raise ValueError(f"unexpected retryable code: {code}")


class GeminiEmbedder:
    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key
        self._fixed_model = model
        self._client: genai.Client | None = None
        self._client_api_key: str | None = None

    def _client_and_model(self) -> tuple[genai.Client, str]:
        s: Settings = get_settings()
        key = (self._api_key or s.gemini_api_key or "").strip()
        if self._client is None or key != self._client_api_key:
            self._client = genai.Client(
                api_key=key if key else None,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._client_api_key = key
        model = self._fixed_model or s.embedding_model
        return self._client, model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client, model = self._client_and_model()
        settings = get_settings()
        last_exc: errors.APIError | None = None
        for attempt in range(settings.embed_retry_max):
            try:
                response = client.models.embed_content(model=model, contents=texts)
                return [list(e.values) for e in response.embeddings]
            except errors.APIError as exc:
                if exc.code not in _RETRYABLE_CODES:
                    raise
                last_exc = exc
                if attempt + 1 >= settings.embed_retry_max:
                    break
                time.sleep(_backoff_seconds(exc.code, attempt, settings, exc))
        assert last_exc is not None
        raise last_exc
