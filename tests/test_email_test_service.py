from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evenor.application.email_test_service import (
    EmailTestError,
    build_dev_inject_smtp_config,
    get_email_test_connectors,
    inject_test_email,
)
from evenor.config.settings import get_settings, reset_settings_cache_for_tests
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType


@pytest.fixture
def email_test_env(tmp_path, monkeypatch):
    db = tmp_path / "email_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    from cryptography.fernet import Fernet

    monkeypatch.setenv("APP_SECRET_KEY", Fernet.generate_key().decode())
    reset_settings_cache_for_tests()
    settings = get_settings()
    from evenor.adapters.persistence.engine import create_db_engine, session_factory
    from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from evenor.application.tenant_service import TenantService

    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="Email Test Bot",
            slug="email-test-bot",
        ).tenant
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "imap_host": "greenmail",
                "imap_port": "3143",
                "username": "bot@test.local",
                "password": "secret",
            },
        )
        session.commit()
        tenant_id = tenant.id
    yield factory, settings, tenant_id
    engine.dispose()
    reset_settings_cache_for_tests()


def test_build_dev_inject_smtp_config_defaults() -> None:
    reset_settings_cache_for_tests()
    settings = get_settings()
    cfg = build_dev_inject_smtp_config(settings)
    assert cfg["smtp_host"] == "greenmail"
    assert cfg["smtp_port"] == "3025"
    assert cfg["smtp_use_tls"] is False


@patch("evenor.application.email_test_service.build_email_sender")
def test_inject_test_email_uses_greenmail_not_out_connector(mock_build, email_test_env) -> None:
    factory, settings, tenant_id = email_test_env
    with factory() as session:
        config_in = get_email_test_connectors(session, tenant_id)
    sender = MagicMock()
    mock_build.return_value = sender

    inject_test_email(
        settings,
        config_in,
        from_addr="client@example.com",
        subject="Hi",
        body="Need a quote",
    )

    mock_build.assert_called_once()
    inject_cfg = mock_build.call_args[0][0]
    assert inject_cfg["smtp_host"] == "greenmail"
    assert inject_cfg["smtp_port"] == "3025"
    sender.send.assert_called_once()
    sent = sender.send.call_args[0][0]
    assert sent.to_addr == "bot@test.local"
    assert sent.from_addr == "client@example.com"


def test_get_email_test_connectors_requires_only_in(email_test_env) -> None:
    factory, _settings, tenant_id = email_test_env
    with factory() as session:
        cfg = get_email_test_connectors(session, tenant_id)
    assert cfg["username"] == "bot@test.local"


def test_get_email_test_connectors_missing_in(email_test_env) -> None:
    factory, _settings, tenant_id = email_test_env
    with factory() as session:
        from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository

        repo = SqlAlchemyConnectorRepository(session)
        row = repo.find_by_tenant_direction_type(
            tenant_id, direction=ConnectorDirection.IN, type=ConnectorType.EMAIL
        )
        assert row is not None
        repo.update(row.id, active=False)
        session.commit()
        with pytest.raises(EmailTestError, match="inbound"):
            get_email_test_connectors(session, tenant_id)
