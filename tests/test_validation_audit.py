from __future__ import annotations

from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from evenor.application.validation_audit_service import ValidationAuditService
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from evenor.domain.models.pending_reply import PendingReplyStatus
from evenor.domain.models.pending_reply_audit import ValidationAuditAction


def test_resolve_reply_logs_audit_and_sets_resolved_fields(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Hello",
        )
        session.commit()
        reply_id = pending.id

    with factory() as session:
        repo = SqlAlchemyPendingReplyRepository(session)
        reply = repo.find_by_id(reply_id)
        assert reply is not None
        ValidationAuditService(session).resolve_reply(
            reply,
            status=PendingReplyStatus.APPROVED,
            actor_email="op@example.com",
        )
        session.commit()

    with factory() as session:
        saved = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        assert saved is not None
        assert saved.status == PendingReplyStatus.APPROVED
        assert saved.resolved_by == "op@example.com"
        assert saved.resolved_at is not None
        activity = ValidationAuditService(session).list_activity(tenant.id, limit=10)
        assert any(e.action == "approved" and e.actor_email == "op@example.com" for e in activity)


def test_list_timeline_includes_attachment_event(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id="email:a@b.com",
            channel="email",
            recipient_id="a@b.com",
            draft_text="Hi",
        )
        session.commit()
        reply_id = pending.id

    with factory() as session:
        ValidationAuditService(session).log_event(
            tenant_id=tenant.id,
            pending_reply_id=reply_id,
            action=ValidationAuditAction.ATTACHMENT_ADDED,
            actor_email="op@example.com",
            detail={"filename": "doc.pdf"},
        )
        session.commit()

    with factory() as session:
        timeline = ValidationAuditService(session).list_timeline_for_reply(tenant.id, reply_id)
        assert len(timeline) == 1
        assert "doc.pdf" in timeline[0].summary
