from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RewriteLanguageGate(Protocol):
    """Decides whether the RAG step may call the LLM to rewrite the user query for retrieval."""

    def allow_llm_rewrite(self, text: str) -> bool:
        ...
