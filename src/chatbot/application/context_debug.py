from __future__ import annotations

import json

from chatbot.domain.models.context_debug import ContextDebugInfo


def format_context_debug_label(debug: ContextDebugInfo | None) -> str:
    if debug is None:
        return ""
    parts = [f"RAG: {debug.rag_chunks} chunks, {_format_chars(debug.rag_chars)}"]
    if debug.customer_chars:
        parts.append(f"Customer: {_format_chars(debug.customer_chars)}")
    parts.append(f"System: {_format_chars(debug.system_chars)}")
    return " · ".join(parts)


def context_debug_to_json(debug: ContextDebugInfo | None) -> str | None:
    if debug is None:
        return None
    return json.dumps(
        {
            "rag_chunks": debug.rag_chunks,
            "rag_chars": debug.rag_chars,
            "customer_chars": debug.customer_chars,
            "system_chars": debug.system_chars,
        },
        ensure_ascii=True,
    )


def context_debug_from_json(raw: str | None) -> ContextDebugInfo | None:
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return ContextDebugInfo(
        rag_chunks=int(data.get("rag_chunks") or 0),
        rag_chars=int(data.get("rag_chars") or 0),
        customer_chars=int(data.get("customer_chars") or 0),
        system_chars=int(data.get("system_chars") or 0),
    )


def _format_chars(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k chars"
    return f"{n} chars"
