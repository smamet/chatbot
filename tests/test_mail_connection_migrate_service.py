from __future__ import annotations

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.application.connector_service import ConnectorService
from chatbot.application.mail_connection_migrate_service import MailConnectionMigrateService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType


def test_mail_connection_migrate_groups_in_and_out(test_settings, test_tenant) -> None:
    tenant, _slug = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        svc.upsert(
            tenant_id=tenant.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.DIRECT,
            config={
                "auth_type": "microsoft_oauth",
                "microsoft_client_id": "cid",
                "microsoft_client_secret": "sec",
                "username": "bot@example.com",
                "oauth_refresh_token": "rt-in",
                "imap_host": "outlook.office365.com",
            },
            active=True,
        )
        svc.upsert(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.DIRECT,
            config={
                "auth_type": "microsoft_oauth",
                "microsoft_client_id": "cid",
                "microsoft_client_secret": "sec",
                "smtp_username": "bot@example.com",
                "outbound_provider": "smtp",
                "from_addr": "bot@example.com",
            },
            active=True,
        )
        session.flush()
        result = MailConnectionMigrateService(session).migrate_tenant(tenant.id)
        session.flush()
        assert result.connections_created == 1
        assert result.connectors_updated == 2
        in_conn = svc.find(tenant.id, direction=ConnectorDirection.IN, type=ConnectorType.EMAIL)
        out_conn = svc.find(tenant.id, direction=ConnectorDirection.OUT, type=ConnectorType.EMAIL)
        assert in_conn is not None and out_conn is not None
        assert in_conn.config.get("mail_connection_id") == out_conn.config.get("mail_connection_id")
        assert "oauth_refresh_token" not in in_conn.config
