from __future__ import annotations

from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType


def test_update_quote_fields_clears_fulfillment_error(test_settings, test_tenant) -> None:
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
            draft_text="Draft",
        )
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            pending.id,
            fulfillment_error="Submit failed",
        )
        session.commit()
        reply_id = pending.id

    with factory() as session:
        repo = SqlAlchemyPendingReplyRepository(session)
        assert repo.find_by_id(reply_id).fulfillment_error == "Submit failed"
        repo.update_quote_fields(reply_id, fulfillment_error=None)
        session.commit()

    with factory() as session:
        assert SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id).fulfillment_error is None


def test_update_quote_fields_sets_quote_erp_modified(test_settings, test_tenant) -> None:
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
            draft_text="Draft",
        )
        session.commit()
        reply_id = pending.id

    with factory() as session:
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            reply_id,
            quote_erp_modified="2026-06-15 14:17:39",
        )
        session.commit()

    with factory() as session:
        assert (
            SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id).quote_erp_modified
            == "2026-06-15 14:17:39"
        )
