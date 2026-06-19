from __future__ import annotations

from datetime import UTC, datetime

from chatbot.application.mail_connection_service import MailConnectionService
from chatbot.domain.models.mail_connection import MailConnection, MailConnectionProvider
from chatbot.domain.models.mail_connection_presets import build_runtime_mail_config


def _connection(**overrides) -> MailConnection:
    base = {
        "id": 1,
        "tenant_id": 10,
        "label": "Support",
        "provider": MailConnectionProvider.MICROSOFT_OAUTH,
        "mailbox_email": "support@example.com",
        "config": {
            "microsoft_client_id": "cid",
            "microsoft_client_secret": "sec",
            "oauth_refresh_token": "rt",
        },
        "active": True,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    base.update(overrides)
    return MailConnection(**base)


def test_build_runtime_mail_config_in_microsoft() -> None:
    cfg = build_runtime_mail_config(_connection(), direction="in")
    assert cfg["imap_host"] == "outlook.office365.com"
    assert cfg["imap_port"] == "993"
    assert cfg["username"] == "support@example.com"
    assert cfg["auth_type"] == "microsoft_oauth"
    assert cfg["oauth_refresh_token"] == "rt"


def test_build_runtime_mail_config_out_google() -> None:
    conn = _connection(
        provider=MailConnectionProvider.GOOGLE_OAUTH,
        config={"google_client_id": "g", "oauth_refresh_token": "rt"},
    )
    cfg = build_runtime_mail_config(conn, direction="out")
    assert cfg["smtp_host"] == "smtp.gmail.com"
    assert cfg["smtp_port"] == "587"
    assert cfg["smtp_username"] == "support@example.com"
    assert cfg["auth_type"] == "google_oauth"


def test_resolve_for_connector_merges_connector_overlay(test_settings, test_tenant) -> None:
    from unittest.mock import patch

    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.engine import create_db_engine, session_factory
    from chatbot.application.connector_service import ConnectorService
    from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType

    tenant, _slug = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        svc = MailConnectionService(session)
        conn = svc.upsert(
            tenant_id=tenant.id,
            connection_id=None,
            label="Sales",
            provider="microsoft_oauth",
            mailbox_email="sales@vdtec.net",
            config_incoming={"microsoft_client_id": "cid", "oauth_refresh_token": "rt"},
        )
        connector = ConnectorService(SqlAlchemyConnectorRepository(session)).upsert(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "mail_connection_id": conn.id,
                "auth_type": "microsoft_oauth",
                "from_addr": "sales@vdtec.net",
                "default_subject": "Reply from support",
                "outbound_provider": "smtp",
            },
            active=True,
        )
        session.flush()
        runtime = {
            "auth_type": "microsoft_oauth",
            "smtp_host": "smtp.office365.com",
            "smtp_username": "sales@vdtec.net",
        }
        with patch.object(svc, "resolve_runtime_config", return_value=(runtime, None)):
            resolved = svc.resolve_for_connector(connector, direction="out", settings=test_settings)
        assert resolved["from_addr"] == "sales@vdtec.net"
        assert resolved["default_subject"] == "Reply from support"
        assert resolved["smtp_username"] == "sales@vdtec.net"
        assert resolved["smtp_host"] == "smtp.office365.com"


def test_resolve_for_connector_ignores_stale_runtime_token_on_connector(
    test_settings, test_tenant
) -> None:
    from unittest.mock import patch

    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.engine import create_db_engine, session_factory
    from chatbot.application.connector_service import ConnectorService
    from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType

    tenant, _slug = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        svc = MailConnectionService(session)
        conn = svc.upsert(
            tenant_id=tenant.id,
            connection_id=None,
            label="Sales",
            provider="microsoft_oauth",
            mailbox_email="sales@vdtec.net",
            config_incoming={"microsoft_client_id": "cid", "oauth_refresh_token": "rt"},
        )
        connector = ConnectorService(SqlAlchemyConnectorRepository(session)).upsert(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "mail_connection_id": conn.id,
                "auth_type": "microsoft_oauth",
                "from_addr": "sales@vdtec.net",
                "outbound_provider": "smtp",
                "_resolved_access_token": "",
            },
            active=True,
        )
        session.flush()
        runtime = {
            "auth_type": "microsoft_oauth",
            "smtp_host": "smtp.office365.com",
            "smtp_username": "sales@vdtec.net",
            "_resolved_access_token": "fresh-token",
        }
        with patch.object(svc, "resolve_runtime_config", return_value=(runtime, None)):
            resolved = svc.resolve_for_connector(connector, direction="out", settings=test_settings)
        assert resolved["_resolved_access_token"] == "fresh-token"


def test_mail_connection_service_upsert_and_delete_guard(test_settings, test_tenant) -> None:
    import pytest

    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.engine import create_db_engine, session_factory
    from chatbot.application.connector_service import ConnectorService
    from chatbot.application.mail_connection_service import MailConnectionError
    from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType

    tenant, _slug = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        svc = MailConnectionService(session)
        conn = svc.upsert(
            tenant_id=tenant.id,
            connection_id=None,
            label="M365",
            provider="microsoft_oauth",
            mailbox_email="a@example.com",
            config_incoming={"microsoft_client_id": "cid", "microsoft_client_secret": "sec"},
        )
        assert conn.id > 0
        ConnectorService(SqlAlchemyConnectorRepository(session)).upsert(
            tenant_id=tenant.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.DIRECT,
            config={"mail_connection_id": conn.id, "auth_type": "microsoft_oauth"},
            active=True,
        )
        session.flush()
        with pytest.raises(MailConnectionError, match="Cannot delete"):
            svc.delete(tenant.id, conn.id)
