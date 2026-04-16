from __future__ import annotations

from google import genai

from chatbot.domain.contracts.embedder import Embedder


class GeminiEmbedder:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.models.embed_content(model=self._model, contents=texts)
        return [list(e.values) for e in response.embeddings]
