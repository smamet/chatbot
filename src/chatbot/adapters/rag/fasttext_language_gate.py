from __future__ import annotations

import logging
from pathlib import Path

from chatbot.adapters.rag.rewrite_language_decision import explain_allow_llm_rewrite_for_rag
from chatbot.config.settings import Settings
from chatbot.domain.contracts.rewrite_language_gate import RewriteLanguageGate

logger = logging.getLogger(__name__)


def _normalize_labels(raw: object) -> list[str]:
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            s = item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
            out.append(s.replace("__label__", "").strip().lower())
    return out


def _normalize_probs(raw: object) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return []


class FastTextRewriteLanguageGate:
    """Uses Facebook fastText `lid.176` (or compatible) to gate the Gemini retrieval rewrite."""

    def __init__(self, settings: Settings) -> None:
        import fasttext  # noqa: PLC0415 — heavy native dep, load only when this class is used

        path = settings.fasttext_lid_model_path
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"FASTTEXT_LID_MODEL_PATH must point to an existing lid file: {path!r}")
        self._settings = settings
        self._model = fasttext.load_model(str(path))
        codes = [x.strip().lower() for x in settings.rag_rewrite_creole_labels.split(",") if x.strip()]
        self._creole = frozenset(codes)

    def allow_llm_rewrite(self, text: str) -> bool:
        line = text.replace("\n", " ").strip()
        if len(line) < 2:
            if self._settings.rag_verbose:
                logger.info("[RAG][lid] text_too_short -> allow_rewrite=False (len=%s)", len(line))
            return False
        labels, probs = self._model.predict(line, k=3)
        lid_labels = _normalize_labels(labels)
        ps = _normalize_probs(probs)
        allowed, reason = explain_allow_llm_rewrite_for_rag(
            lid_top_labels=lid_labels,
            lid_probs=ps,
            min_prob_en=self._settings.rag_rewrite_min_prob_en,
            fr_max_prob_creole=self._settings.rag_rewrite_fr_max_prob_creole,
            creole_labels=self._creole,
        )
        if self._settings.rag_verbose:
            pairs = list(zip(lid_labels, ps, strict=False))
            logger.info(
                "[RAG][lid] fasttext_top3=%s creole_labels=%s min_prob_en=%s fr_creole_below=%s -> allow_rewrite=%s (%s)",
                pairs,
                sorted(self._creole),
                self._settings.rag_rewrite_min_prob_en,
                self._settings.rag_rewrite_fr_max_prob_creole,
                allowed,
                reason,
            )
        return allowed


def load_rewrite_language_gate(settings: Settings) -> RewriteLanguageGate | None:
    """If lang filter is off or model path is missing, return None (no gating)."""
    if not settings.rag_rewrite_lang_filter:
        if settings.rag_verbose:
            logger.info("[RAG][lid] lang_filter=OFF -> no FastText gate (rewrite only controlled by RAG_REWRITE_ENABLED)")
        return None
    path = settings.fasttext_lid_model_path
    if path is None or not Path(path).is_file():
        logger.warning(
            "RAG_REWRITE_LANG_FILTER is enabled but FASTTEXT_LID_MODEL_PATH is missing or not a file; "
            "LLM query rewrite will not be filtered by language (all rewrites allowed when RAG_REWRITE_ENABLED)."
        )
        if settings.rag_verbose:
            logger.info("[RAG][lid] lang_filter=ON but model_path missing -> gate=None (no lid filtering)")
        return None
    if settings.rag_verbose:
        logger.info("[RAG][lid] lang_filter=ON model_path=%s -> FastText gate loaded", path)
    return FastTextRewriteLanguageGate(settings)
