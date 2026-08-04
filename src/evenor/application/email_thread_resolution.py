from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

METHOD_LABELS: dict[str, str] = {
    "rfc_headers": "RFC headers (In-Reply-To / References)",
    "subject_exact": "exact subject match",
    "subject_similarity": "subject similarity",
    "llm": "LLM disambiguation",
    "thread_key_reuse": "existing thread key",
    "new_thread": "new thread",
}


@dataclass(frozen=True, slots=True)
class ThreadResolutionLlmMeta:
    confidence: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ThreadResolutionAudit:
    method: str
    used_llm: bool
    steps: tuple[str, ...]
    llm: ThreadResolutionLlmMeta | None = None


def audit_to_json(audit: ThreadResolutionAudit) -> str:
    payload: dict[str, Any] = {
        "method": audit.method,
        "used_llm": audit.used_llm,
        "steps": list(audit.steps),
        "llm": None,
    }
    if audit.llm is not None:
        payload["llm"] = {
            "confidence": audit.llm.confidence,
            "prompt_tokens": audit.llm.prompt_tokens,
            "output_tokens": audit.llm.output_tokens,
        }
    return json.dumps(payload, separators=(",", ":"))


def audit_from_json(raw: str | None) -> ThreadResolutionAudit | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    method = str(data.get("method") or "").strip()
    if not method:
        return None
    steps_raw = data.get("steps") or []
    steps = tuple(str(s) for s in steps_raw) if isinstance(steps_raw, list) else ()
    llm_raw = data.get("llm")
    llm: ThreadResolutionLlmMeta | None = None
    if isinstance(llm_raw, dict):
        llm = ThreadResolutionLlmMeta(
            confidence=_optional_float(llm_raw.get("confidence")),
            prompt_tokens=_optional_int(llm_raw.get("prompt_tokens")),
            output_tokens=_optional_int(llm_raw.get("output_tokens")),
        )
    return ThreadResolutionAudit(
        method=method,
        used_llm=bool(data.get("used_llm")),
        steps=steps,
        llm=llm,
    )


def format_resolution_tooltip(audit: ThreadResolutionAudit) -> str:
    if audit.used_llm and audit.llm is not None:
        parts = ["LLM disambiguation"]
        token_bits: list[str] = []
        if audit.llm.prompt_tokens is not None:
            token_bits.append(f"{audit.llm.prompt_tokens} in")
        if audit.llm.output_tokens is not None:
            token_bits.append(f"{audit.llm.output_tokens} out")
        if token_bits:
            parts.append(f"({' / '.join(token_bits)} tokens)")
        if audit.llm.confidence is not None:
            parts.append(f"confidence {audit.llm.confidence:.2f}")
        fallback = METHOD_LABELS.get(audit.method, audit.method)
        if audit.method != "llm":
            parts.append(f"→ {fallback}")
        return " ".join(parts)

    label = METHOD_LABELS.get(audit.method, audit.method)
    if len(audit.steps) <= 1:
        return f"Resolved via: {label}"
    readable = " → ".join(_step_label(step) for step in audit.steps)
    return f"Steps: {readable}"


def _step_label(step: str) -> str:
    if step.endswith(":hit"):
        base = step[:-4]
        return METHOD_LABELS.get(base, base.replace("_", " "))
    if step.endswith(":miss"):
        base = step[:-5]
        return f"{METHOD_LABELS.get(base, base)} miss"
    if step == "subject_ambiguous":
        return "ambiguous subject"
    if step.startswith("llm:"):
        suffix = step[4:]
        if suffix == "low_confidence":
            return "LLM low confidence"
        if suffix == "miss":
            return "LLM no match"
        return "LLM"
    return step.replace("_", " ")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
