from __future__ import annotations

"""Pure rules for when FastText lid allows an LLM retrieval rewrite (English, Creole ISO codes, or low-confidence French)."""


def explain_allow_llm_rewrite_for_rag(
    *,
    lid_top_labels: list[str],
    lid_probs: list[float],
    min_prob_en: float,
    fr_max_prob_creole: float,
    creole_labels: frozenset[str],
) -> tuple[bool, str]:
    """
    Return (allow_llm_rewrite, short_reason) for logging and tests.

    Allow LLM rewrite only for:
    - English with probability >= min_prob_en
    - Any label in creole_labels when it is the top prediction
    - French with probability < fr_max_prob_creole (Creole often misclassified as fr with low score)
    """
    if not lid_top_labels or not lid_probs:
        return False, "no_lid_output"
    top = lid_top_labels[0].strip().lower()
    p0 = float(lid_probs[0])
    if top == "en" and p0 >= min_prob_en:
        return True, f"english_top(p={p0:.4f}>={min_prob_en})"
    if top == "en":
        return False, f"english_low_confidence(p={p0:.4f}<{min_prob_en})"
    if top in creole_labels:
        return True, f"creole_top_label={top}(p={p0:.4f})"
    if top == "fr" and p0 < fr_max_prob_creole:
        return True, f"french_low_conf_as_creole_proxy(p={p0:.4f}<{fr_max_prob_creole})"
    if top == "fr":
        return False, f"french_high_conf_no_rewrite(p={p0:.4f}>={fr_max_prob_creole})"
    return False, f"other_language_top={top}(p={p0:.4f})"


def should_allow_llm_rewrite_for_rag(
    *,
    lid_top_labels: list[str],
    lid_probs: list[float],
    min_prob_en: float,
    fr_max_prob_creole: float,
    creole_labels: frozenset[str],
) -> bool:
    return explain_allow_llm_rewrite_for_rag(
        lid_top_labels=lid_top_labels,
        lid_probs=lid_probs,
        min_prob_en=min_prob_en,
        fr_max_prob_creole=fr_max_prob_creole,
        creole_labels=creole_labels,
    )[0]
