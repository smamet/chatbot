from __future__ import annotations

from chatbot.domain.contracts.llm_client import LlmUsage


def usage_from_response(response: object) -> LlmUsage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return LlmUsage()
    return LlmUsage(
        prompt_tokens=getattr(meta, "prompt_token_count", None),
        candidates_tokens=getattr(meta, "candidates_token_count", None),
        total_tokens=getattr(meta, "total_token_count", None),
    )
