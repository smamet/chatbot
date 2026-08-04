from __future__ import annotations


def estimate_text_tokens(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def reduction_percent(raw_tokens: int, new_tokens: int) -> int | None:
    if raw_tokens <= 0:
        return None
    pct = round((1 - (new_tokens / raw_tokens)) * 100)
    return max(0, min(100, pct))
