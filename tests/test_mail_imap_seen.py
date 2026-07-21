from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.channel_outbound import approve_pending_reply
from chatbot.application.mail_imap_seen_service import mark_imap_seen_for_pending_reply
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus


def _pending_reply(**overrides) -> PendingReply:
    now = datetime.now(UTC)
    base = {
        "id": 1,
        "tenant_id": 10,
        "connector_id": 2,
        "session_id": "email:client@example.com",
        "channel": ConnectorType.EMAIL.value,
        "recipient_id": "client@example.com",
        "draft_text": "Bot reply",
        "status": PendingReplyStatus.PENDING,
        "created_at": now,
        "updated_at": now,
        "fulfillment_kind": FulfillmentKind.REPLY_ONLY,
    }
    base.update(overrides)
    return PendingReply(**base)


def _email_connectors(session, tenant_id: int) -> tuple[int, int]:
    repo = SqlAlchemyConnectorRepository(session)
    in_conn = repo.create(
        tenant_id=tenant_id,
        direction=ConnectorDirection.IN,
        type=ConnectorType.EMAIL,
        mode=ConnectorMode.DIRECT,
        config={
            "imap_host": "greenmail",
            "imap_port": "3143",
            "username": "bot@test.local",
            "password": "secret",
        },
    )
    out_conn = repo.create(
        tenant_id=tenant_id,
        direction=ConnectorDirection.OUT,
        type=ConnectorType.EMAIL,
        mode=ConnectorMode.VALIDATION,
        config={
            "outbound_provider": "smtp",
            "smtp_host": "mailpit",
            "smtp_port": "1025",
            "from_addr": "bot@test.local",
        },
    )
    session.flush()
    return in_conn.id, out_conn.id


@patch("chatbot.application.mail_imap_seen_service.imap_client")
@patch("chatbot.application.mail_imap_seen_service.prepare_email_connector_config")
def test_mark_imap_seen_for_pending_reply(mock_prepare, mock_imap_ctx, test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    imap = MagicMock()
    mock_imap_ctx.return_value.__enter__.return_value = imap
    mock_prepare.return_value = {"imap_host": "greenmail"}

    with factory() as session:
        _email_connectors(session, tenant.id)
        draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant.id)
        draft = draft_repo.create(
            imap_uid="42",
            from_addr="client@example.com",
            to_addr="bot@test.local",
            subject="Hello",
            body_in="Question",
        )
        draft_repo.mark_processed(draft.id, draft_reply="Bot reply")
        session.commit()

    with factory() as session:
        reply = _pending_reply(tenant_id=tenant.id, draft_text="Edited reply text")
        mark_imap_seen_for_pending_reply(
            session,
            tenant_id=tenant.id,
            reply=reply,
            settings=test_settings,
        )

    imap.mark_seen.assert_called_once_with("42")


@patch("chatbot.application.channel_outbound.mark_imap_seen_for_pending_reply")
@patch("chatbot.application.channel_outbound.dispatch_channel_reply")
def test_approve_pending_reply_marks_imap_seen(mock_dispatch, mock_mark_seen, test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    with factory() as session:
        _, out_id = _email_connectors(session, tenant.id)
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=out_id,
            session_id="email:client@example.com",
            channel=ConnectorType.EMAIL.value,
            recipient_id="client@example.com",
            draft_text="Hello",
        )
        session.commit()
        reply_id = pending.id

    with factory() as session:
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        assert reply is not None
        approve_pending_reply(
            session,
            reply,
            config={"from_addr": "bot@test.local"},
            settings=test_settings,
        )

    mock_dispatch.assert_called_once()
    mock_mark_seen.assert_called_once()
