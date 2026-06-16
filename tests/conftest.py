from __future__ import annotations

import uuid

import pytest
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.tenant_service import TenantService
from chatbot.config.settings import Settings, reset_settings_cache_for_tests
from chatbot.domain.models.tenant import Tenant, TenantConfig


class TestSettings(Settings):
    """App settings for tests: explicit kwargs only (never OS env for DATABASE_URL, etc.)."""

    __test__ = False

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
    from cryptography.fernet import Fernet

    data = tmp_path / "data"
    db_name = f"test_{uuid.uuid4().hex}.db"
    return TestSettings(
        gemini_api_key="test-key",
        admin_token="test-admin",
        app_secret_key=Fernet.generate_key().decode(),
        session_secret="test-session",
        database_url=f"sqlite:///{data / db_name}",
        data_root=data,
        lancedb_root=data / "lancedb",
        rag_enabled=False,
    )


@pytest.fixture
def test_tenant(test_settings: TestSettings) -> tuple[Tenant, str]:
    """Create one tenant in DB; yields (tenant, plaintext_token)."""
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        svc = TenantService(SqlAlchemyTenantRepository(session))
        result = svc.create_tenant(
            name="Test Tenant",
            slug="test",
            prompt="You are a test bot.",
            config=TenantConfig(rag_enabled=False),
        )
        session.commit()
        yield result.tenant, result.token
    finally:
        session.close()
        engine.dispose()
