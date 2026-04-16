from __future__ import annotations

from pathlib import Path

from chatbot.adapters.rag.parsers.registry import parse_file, supported_suffixes


def test_parse_markdown_roundtrip(tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("# Title\n\nParagraph **bold**.", encoding="utf-8")
    assert parse_file(doc) == "# Title\n\nParagraph **bold**."


def test_md_in_supported_suffixes() -> None:
    assert ".md" in supported_suffixes()
