from __future__ import annotations

import uuid

import pytest
from pydantic_settings import SettingsConfigDict

from chatbot.config.settings import Settings, get_settings


class TestSettings(Settings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


@pytest.fixture(autouse=True)
def _dummy_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key-for-tests")


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(tmp_path) -> TestSettings:
    db_name = f"test_{uuid.uuid4().hex}.db"
    return TestSettings(
        gemini_api_key="test-key",
        database_url=f"sqlite:///{tmp_path / db_name}",
        lancedb_path=tmp_path / "lancedb",
        prompt_path=tmp_path / "prompt.md",
        rag_enabled=False,
    )
