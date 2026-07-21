from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextDebugInfo:
    rag_chunks: int = 0
    rag_chars: int = 0
    customer_chars: int = 0
    system_chars: int = 0
