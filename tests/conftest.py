from __future__ import annotations

import uuid

import pytest
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from chatbot.config.settings import Settings, reset_settings_cache_for_tests


class TestSettings(Settings):
    """App settings for tests: explicit kwargs only (never OS env for DATABASE_URL, etc.)."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", populate_by_name=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


@pytest.fixture(autouse=True)
def _dummy_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key-for-tests")


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    reset_settings_cache_for_tests()
    yield
    reset_settings_cache_for_tests()


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
