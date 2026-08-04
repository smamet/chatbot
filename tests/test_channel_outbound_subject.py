from __future__ import annotations

from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.application.channel_outbound import queue_pending_reply
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType


def test_queue_pending_reply_prefills_re_subject(test_settings) -> None:
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
        from evenor.application.tenant_service import TenantService
        from evenor.domain.models.tenant import TenantConfig

        result = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="Subject Bot",
            slug="subject-bot",
            prompt="Test",
            config=TenantConfig(rag_enabled=False),
        )
        tenant_id = result.tenant.id
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        draft_text = "Thanks for your message"
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).create(
            imap_uid="uid-2",
            from_addr="client@example.com",
            to_addr="bot@test.local",
            subject="Need a quote",
            body_in="Please quote",
        )
        SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).mark_processed(
            draft.id, draft_reply=draft_text
        )
        pending = queue_pending_reply(
            session,
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text=draft_text,
        )
        assert pending.draft_subject == "Re: Need a quote"
        session.commit()
    engine.dispose()
