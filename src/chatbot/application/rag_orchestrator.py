from __future__ import annotations

from chatbot.config.settings import Settings
from chatbot.domain.contracts.embedder import Embedder
from chatbot.domain.contracts.llm_client import LlmClient
from chatbot.domain.contracts.vector_store import VectorStore
from chatbot.domain.models.message import ChatMessage, MessageRole


class RagPipeline:
    """Rewrite query (optional) → embed → retrieve → format context for the main LLM."""

    def __init__(
        self,
        *,
        settings: Settings,
        rewriter_llm: LlmClient,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._rewriter_llm = rewriter_llm
        self._embedder = embedder
        self._store = vector_store

    def build_retrieval_context(self, user_query: str) -> str:
        if not self._settings.rag_enabled:
            return ""
        q = user_query.strip()
        if not q:
            return ""
        search_query = self._rewrite_query(q) if self._settings.rag_rewrite_enabled else q
        vectors = self._embedder.embed_texts([search_query])
        if not vectors:
            return ""
        hits = self._store.search(vectors[0], top_k=self._settings.rag_top_k)
        if not hits:
            return ""
        lines: list[str] = []
        for h in hits:
            lines.append(f"[{h.source_path} | chunk {h.chunk_id}]\n{h.text}")
        return "\n\n---\n\n".join(lines)

    def _rewrite_query(self, user_query: str) -> str:
        lang = self._settings.retrieval_language
        system = (
            f"Rewrite the user's message into a short keyword-rich search query in "
            f"language code '{lang}' for document retrieval. Reply with ONLY the query text, no quotes."
        )
        result = self._rewriter_llm.generate_chat(
            system_instruction=system,
            messages=[ChatMessage(role=MessageRole.USER, content=user_query)],
        )
        out = result.text.strip() or user_query
        return out
