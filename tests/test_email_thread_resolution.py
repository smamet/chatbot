from __future__ import annotations

from chatbot.application.email_thread_resolution import (
    ThreadResolutionAudit,
    ThreadResolutionLlmMeta,
    audit_from_json,
    audit_to_json,
    format_resolution_tooltip,
)


def test_audit_round_trip() -> None:
    audit = ThreadResolutionAudit(
        method="llm",
        used_llm=True,
        steps=("subject_ambiguous", "llm"),
        llm=ThreadResolutionLlmMeta(
            confidence=0.92,
            prompt_tokens=180,
            output_tokens=24,
        ),
    )
    restored = audit_from_json(audit_to_json(audit))
    assert restored == audit


def test_format_tooltip_without_llm() -> None:
    audit = ThreadResolutionAudit(
        method="rfc_headers",
        used_llm=False,
        steps=("rfc_headers",),
    )
    assert format_resolution_tooltip(audit) == "Resolved via: RFC headers (In-Reply-To / References)"


def test_format_tooltip_with_llm() -> None:
    audit = ThreadResolutionAudit(
        method="llm",
        used_llm=True,
        steps=("subject_ambiguous", "llm"),
        llm=ThreadResolutionLlmMeta(confidence=0.91, prompt_tokens=100, output_tokens=12),
    )
    text = format_resolution_tooltip(audit)
    assert "LLM disambiguation" in text
    assert "100 in" in text
    assert "12 out" in text
    assert "0.91" in text


def test_audit_from_json_invalid_returns_none() -> None:
    assert audit_from_json("") is None
    assert audit_from_json("{bad") is None
