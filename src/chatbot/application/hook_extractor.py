from __future__ import annotations

import json
from dataclasses import dataclass

from chatbot.domain.constants import HOOK_MARKER, LEGACY_HOOK_MARKER


@dataclass(frozen=True, slots=True)
class ExtractedHook:
    clean_reply: str
    hook_type: str | None
    payload_json: str | None


def _find_marker(text: str) -> tuple[int, str] | None:
    for marker in (HOOK_MARKER, LEGACY_HOOK_MARKER):
        idx = text.find(marker)
        if idx >= 0:
            return idx, marker
    return None


def extract_hook(text: str) -> ExtractedHook:
    found = _find_marker(text)
    if found is None:
        return ExtractedHook(clean_reply=text.strip(), hook_type=None, payload_json=None)
    idx, marker = found
    clean_reply = text[:idx].strip()
    payload_str = text[idx + len(marker) :].strip()
    if not payload_str:
        return ExtractedHook(clean_reply=clean_reply, hook_type=None, payload_json=None)
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(payload_str)
    except json.JSONDecodeError:
        return ExtractedHook(clean_reply=clean_reply, hook_type=None, payload_json=None)
    if not isinstance(payload, dict):
        return ExtractedHook(clean_reply=clean_reply, hook_type=None, payload_json=None)
    hook_type = payload.get("type")
    if isinstance(hook_type, str) and hook_type.strip():
        hook_type = hook_type.strip()
    elif "action" in payload:
        hook_type = "order"
    else:
        return ExtractedHook(clean_reply=clean_reply, hook_type=None, payload_json=None)
    payload_json = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return ExtractedHook(clean_reply=clean_reply, hook_type=hook_type, payload_json=payload_json)
