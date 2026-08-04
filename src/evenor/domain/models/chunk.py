from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    text: str
    source_path: str
