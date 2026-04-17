"""RAG rewrite language gate: Creole token markers only (no fastText)."""

from __future__ import annotations

import logging

from chatbot.adapters.rag.creole_script_heuristic import creole_markers_hit
from chatbot.config.settings import Settings
from chatbot.domain.contracts.rewrite_language_gate import RewriteLanguageGate

logger = logging.getLogger(__name__)


class CreoleMarkersRewriteLanguageGate:
    """Allow LLM retrieval rewrite when Mauritian Creole marker tokens are present."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def allow_llm_rewrite(self, text: str) -> bool:
        line = text.replace("\n", " ").strip()
        if len(line) < 2:
            if self._settings.rag_verbose:
                logger.info("[RAG][rewrite_gate] text_too_short -> allow_rewrite=False (len=%s)", len(line))
            return False
        hit = creole_markers_hit(line)
        if self._settings.rag_verbose:
            logger.info("[RAG][rewrite_gate] creole_markers_hit=%s -> allow_rewrite=%s", hit, hit)
        return hit


def load_rewrite_language_gate(settings: Settings) -> RewriteLanguageGate | None:
    """If lang filter is off, return None (no gating). Otherwise marker-only gate."""
    if not settings.rag_rewrite_lang_filter:
        if settings.rag_verbose:
            logger.info("[RAG][rewrite_gate] lang_filter=OFF -> no marker gate")
        return None
    if settings.rag_verbose:
        logger.info("[RAG][rewrite_gate] Creole marker gate active (no fastText)")
    return CreoleMarkersRewriteLanguageGate(settings)
