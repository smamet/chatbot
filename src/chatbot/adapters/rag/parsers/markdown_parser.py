from __future__ import annotations

from pathlib import Path


def parse_markdown(path: Path) -> str:
    """Load markdown as plain text for chunking and embedding."""
    return path.read_text(encoding="utf-8", errors="replace")
