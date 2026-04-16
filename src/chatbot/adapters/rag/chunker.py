from __future__ import annotations

import hashlib

from chatbot.domain.models.chunk import TextChunk


def chunk_text(
    text: str,
    *,
    source_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    text = text.strip()
    if not text:
        return []
    chunks: list[TextChunk] = []
    start = 0
    n = len(text)
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            digest = hashlib.sha256(f"{source_path}:{idx}:{piece[:64]}".encode()).hexdigest()[:16]
            chunk_id = f"{digest}"
            chunks.append(TextChunk(chunk_id=chunk_id, text=piece, source_path=source_path))
            idx += 1
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks
