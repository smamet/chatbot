from __future__ import annotations

import pytest

from chatbot.adapters.rag.rewrite_language_decision import should_allow_llm_rewrite_for_rag

_CREOLE = frozenset({"ht", "gcf"})


@pytest.mark.parametrize(
    ("labels", "probs", "expected"),
    [
        ([], [], False),
        (["en"], [0.9], True),
        (["en"], [0.10], False),
        (["ht"], [0.12], True),
        (["fr"], [0.30], True),
        (["fr"], [0.60], False),
        (["fr"], [0.50], False),
        (["es"], [0.99], False),
        (["fr", "ht"], [0.45, 0.20], True),
    ],
)
def test_should_allow_llm_rewrite_for_rag(labels: list[str], probs: list[float], expected: bool) -> None:
    assert (
        should_allow_llm_rewrite_for_rag(
            lid_top_labels=labels,
            lid_probs=probs,
            min_prob_en=0.20,
            fr_max_prob_creole=0.50,
            creole_labels=_CREOLE,
        )
        == expected
    )
