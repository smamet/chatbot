from __future__ import annotations

from chatbot.config.settings import get_settings, reset_settings_cache_for_tests


def test_email_thread_llm_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("EMAIL_THREAD_LLM_ENABLED", raising=False)
    reset_settings_cache_for_tests()
    assert get_settings().email_thread_llm_enabled is False
    reset_settings_cache_for_tests()
