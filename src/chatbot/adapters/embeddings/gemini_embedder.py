from __future__ import annotations

from google import genai

from chatbot.config.settings import Settings, get_settings
from chatbot.domain.contracts.embedder import Embedder


class GeminiEmbedder:
    """Reads embedding model and API key from ``get_settings()`` on each call."""

    def __init__(self) -> None:
        self._client: genai.Client | None = None
        self._client_api_key: str | None = None

    def _client_and_model(self) -> tuple[genai.Client, str]:
        s: Settings = get_settings()
        key = s.gemini_api_key or ""
        if self._client is None or key != self._client_api_key:
            self._client = genai.Client(api_key=key) if key else genai.Client()
            self._client_api_key = key
        return self._client, s.embedding_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client, model = self._client_and_model()
        response = client.models.embed_content(model=model, contents=texts)
        return [list(e.values) for e in response.embeddings]
