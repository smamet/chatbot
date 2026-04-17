from __future__ import annotations

import pytest

from chatbot.adapters.rag.creole_script_heuristic import creole_markers_hit, looks_like_mauritian_creole_script


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("eki ou fer livraison dans entier moris ?", True),
        ("mo ti pe rode ene parfum pou mo ban airbnb", True),
        ("hi what do you sell exactly?", False),
        ("j'aimerais connaître le prix des diffuseurs.", False),
        ("ski is fun", False),
        ("the ki word alone", True),
    ],
)
def test_creole_markers_hit(text: str, expected: bool) -> None:
    assert creole_markers_hit(text) is expected


def test_looks_like_alias_matches_creole_markers_hit() -> None:
    assert looks_like_mauritian_creole_script("mo fer") is creole_markers_hit("mo fer")
