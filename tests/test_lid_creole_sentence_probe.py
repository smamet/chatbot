"""
Probe for Creole marker rewrite gate (no fastText).

Edit PROBE_SENTENCES below, then run (from repo root):

  pytest tests/test_lid_creole_sentence_probe.py -v -s
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from chatbot.adapters.rag.creole_script_heuristic import creole_markers_hit
from chatbot.adapters.rag.fasttext_language_gate import CreoleMarkersRewriteLanguageGate
from tests.conftest import TestSettings as SettingsForTests

# --- Edit this list to try new sentences ---
PROBE_SENTENCES: list[str] = [
    "hi what do you sell exactly?",
    "j'aimerais connaître le prix des diffuseurs.",
    "eki ou fer livraison dans entier moris ?",
    "mo ti pe rode ene parfum pou mo ban airbnb",
]

EXPECTED_ALLOW_REWRITE: dict[str, bool] = {
    "hi what do you sell exactly?": False,
    "j'aimerais connaître le prix des diffuseurs.": False,
    "eki ou fer livraison dans entier moris ?": True,
    "mo ti pe rode ene parfum pou mo ban airbnb": True,
}


def _probe_settings(tmp_path: Path) -> SettingsForTests:
    (tmp_path / "prompt.md").write_text("test", encoding="utf-8")
    return SettingsForTests(
        gemini_api_key="test-key",
        database_url=f"sqlite:///{tmp_path / f'probe_{uuid.uuid4().hex}.db'}",
        lancedb_path=tmp_path / "lancedb",
        prompt_path=tmp_path / "prompt.md",
        rag_enabled=False,
        rag_rewrite_lang_filter=True,
        rag_verbose=False,
    )


def test_creole_marker_probe_prints_and_matches_expected(tmp_path: Path, capsys) -> None:
    settings = _probe_settings(tmp_path)
    gate = CreoleMarkersRewriteLanguageGate(settings)

    for sentence in PROBE_SENTENCES:
        gate_allow = gate.allow_llm_rewrite(sentence)
        hit = creole_markers_hit(sentence)
        assert gate_allow is hit
        print(f"\n--- {sentence!r}")
        print(f"    creole_markers_hit={hit}  allow_llm_rewrite={gate_allow}")

        if sentence in EXPECTED_ALLOW_REWRITE:
            assert gate_allow is EXPECTED_ALLOW_REWRITE[sentence]

    out = capsys.readouterr().out
    assert "creole_markers_hit=" in out
