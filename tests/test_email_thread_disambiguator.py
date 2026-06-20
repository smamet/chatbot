from __future__ import annotations

from unittest.mock import MagicMock

from chatbot.application.email_thread_disambiguator import EmailThreadDisambiguator


def test_disambiguator_disabled_returns_new_thread() -> None:
    llm = MagicMock()
    result = EmailThreadDisambiguator(llm=llm, enabled=False).disambiguate(
        inbound_subject="devis",
        body_preview="hello",
        candidates=[{"thread_key": "abc", "subject": "devis", "last_activity": "x"}],
    )
    assert result.same_thread is False
    assert result.confidence == 0.0
    assert result.llm_called is False
    llm.generate_chat.assert_not_called()


def test_disambiguator_high_confidence_same_thread() -> None:
    llm = MagicMock()
    llm.generate_chat.return_value = MagicMock(
        text='{"same_thread": true, "confidence": 0.95, "thread_key": "abc123"}',
        usage=MagicMock(prompt_tokens=20, candidates_tokens=6, total_tokens=26),
    )
    result = EmailThreadDisambiguator(llm=llm, min_confidence=0.7).disambiguate(
        inbound_subject="devis",
        body_preview="suite",
        candidates=[{"thread_key": "abc123", "subject": "devis", "last_activity": "x"}],
    )
    assert result.same_thread is True
    assert result.thread_key == "abc123"
    assert result.llm_called is True
    assert result.output_tokens == 6
