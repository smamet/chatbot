from __future__ import annotations

import logging

from chatbot.config.settings import Settings
from chatbot.domain.contracts.embedder import Embedder
from chatbot.domain.contracts.llm_client import LlmClient
from chatbot.domain.contracts.rewrite_language_gate import RewriteLanguageGate
from chatbot.domain.contracts.vector_store import VectorStore
from chatbot.domain.models.message import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


def _preview(text: str, max_len: int = 160) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


class RagPipeline:
    """Rewrite query (optional) → embed → retrieve → format context for the main LLM."""

    def __init__(
        self,
        *,
        settings: Settings,
        rewriter_llm: LlmClient,
        embedder: Embedder,
        vector_store: VectorStore,
        rewrite_language_gate: RewriteLanguageGate | None = None,
    ) -> None:
        self._settings = settings
        self._rewriter_llm = rewriter_llm
        self._embedder = embedder
        self._store = vector_store
        self._rewrite_language_gate = rewrite_language_gate

    def build_retrieval_context(self, user_query: str) -> str:
        v = self._settings.rag_verbose
        if not self._settings.rag_enabled:
            if v:
                logger.info("[RAG] build_retrieval_context skipped: RAG_ENABLED=false")
            return ""
        q = user_query.strip()
        if not q:
            if v:
                logger.info("[RAG] build_retrieval_context skipped: empty user_query")
            return ""
        if v:
            logger.info(
                "[RAG] start user_query_preview=%r len=%s retrieval_language=%s rag_top_k=%s",
                _preview(q, 200),
                len(q),
                self._settings.retrieval_language,
                self._settings.rag_top_k,
            )
            logger.info(
                "[RAG] flags rag_rewrite_enabled=%s lang_gate=%s",
                self._settings.rag_rewrite_enabled,
                type(self._rewrite_language_gate).__name__ if self._rewrite_language_gate else None,
            )

        use_llm_rewrite = self._settings.rag_rewrite_enabled
        if use_llm_rewrite and self._rewrite_language_gate is not None:
            use_llm_rewrite = self._rewrite_language_gate.allow_llm_rewrite(q)
        elif v and use_llm_rewrite and self._rewrite_language_gate is None:
            logger.info("[RAG] rewrite_gate absent -> use_llm_rewrite follows RAG_REWRITE_ENABLED only (%s)", use_llm_rewrite)

        if v:
            logger.info("[RAG] decision use_llm_rewrite=%s (if true, call rewriter for retrieval query)", use_llm_rewrite)

        search_query = self._rewrite_query(q) if use_llm_rewrite else q
        if v:
            logger.info(
                "[RAG] embed_input preview=%r (same_as_user=%s)",
                _preview(search_query, 200),
                search_query.strip() == q,
            )

        vectors = self._embedder.embed_texts([search_query])
        if not vectors:
            if v:
                logger.info("[RAG] embedder returned no vectors -> empty context")
            return ""
        dim = len(vectors[0])
        if v:
            logger.info("[RAG] embedded dim=%s", dim)

        hits = self._store.search(vectors[0], top_k=self._settings.rag_top_k)
        if not hits:
            if v:
                logger.info("[RAG] vector search returned 0 hits -> empty context")
            return ""
        if v:
            for i, h in enumerate(hits):
                logger.info(
                    "[RAG] hit[%s] score=%s chunk_id=%s source=%s text_preview=%r",
                    i,
                    h.score,
                    h.chunk_id,
                    h.source_path,
                    _preview(h.text, 120),
                )
            logger.info("[RAG] done hits=%s context_chars=%s", len(hits), sum(len(x.text) for x in hits))

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
        if self._settings.rag_verbose:
            logger.info(
                "[RAG][rewrite] model_target_language=%r system_instruction_preview=%r user_preview=%r",
                lang,
                _preview(system, 180),
                _preview(user_query, 180),
            )
        result = self._rewriter_llm.generate_chat(
            system_instruction=system,
            messages=[ChatMessage(role=MessageRole.USER, content=user_query)],
        )
        out = result.text.strip() or user_query
        if self._settings.rag_verbose:
            logger.info("[RAG][rewrite] output_preview=%r usage=%s", _preview(out, 200), result.usage)
        return out
