from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from chatbot.__main__ import app
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.mail_imap_uid_repository import SqlAlchemyMailImapUidRepository
from chatbot.adapters.persistence.orm import (
    HookEventRow,
    IngestedFileRow,
    MailDraftRow,
    MailImapUidRow,
    MessageRow,
    OrderRow,
    PendingReplyAuditEventRow,
    PendingReplyEditRow,
    PendingReplyRow,
    TenantRow,
    TestChatSessionRow,
)
from chatbot.adapters.persistence.order_repository import SqlAlchemyOrderRepository
from chatbot.adapters.persistence.pending_reply_audit_repository import (
    SqlAlchemyPendingReplyAuditRepository,
)
from chatbot.adapters.persistence.pending_reply_edit_repository import (
    SqlAlchemyPendingReplyEditRepository,
)
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.adapters.persistence.test_chat_session_repository import TestChatSessionRepository
from chatbot.application.tenant_flush_service import TenantFlushError, TenantFlushService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.integration import IntegrationType
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.domain.models.order import OrderAction, OrderCommand, OrderItem
from chatbot.domain.models.pending_reply_audit import ValidationAuditAction

runner = CliRunner()


def _count(session, model, tenant_id: int) -> int:
    return session.scalar(
        select(func.count()).select_from(model).where(model.tenant_id == tenant_id)
    ) or 0


def _seed_operational_data(test_settings, tenant) -> None:
    slug = tenant.slug
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    att_dir = test_settings.data_root / "attachments" / slug / "1"
    att_dir.mkdir(parents=True, exist_ok=True)
    (att_dir / "file.pdf").write_bytes(b"%PDF")
    quote_dir = test_settings.data_root / "quotes" / slug
    quote_dir.mkdir(parents=True, exist_ok=True)
    (quote_dir / "QTN-0001.pdf").write_bytes(b"%PDF")
    docs_dir = test_settings.data_root / "docs" / slug
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "manual.md").write_text("# Manual", encoding="utf-8")

    with factory() as session:
        SqlAlchemyConversationRepository(session, tenant.id).append_message(
            "email:client@example.com",
            ChatMessage(role=MessageRole.USER, content="Hello"),
        )
        SqlAlchemyHookEventRepository(session, tenant.id).create(
            session_id="email:client@example.com",
            hook_type="order.create",
            payload_json="{}",
        )
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        SqlAlchemyIntegrationRepository(session).create(
            tenant_id=tenant.id,
            type=IntegrationType.ERPNEXT,
            config={"url": "https://erp.example.com"},
            active=True,
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Draft",
        )
        SqlAlchemyPendingReplyEditRepository(session).create(
            tenant_id=tenant.id,
            pending_reply_id=pending.id,
            edited_by="admin@example.com",
            body_before="a",
            body_after="b",
            diff="d",
        )
        SqlAlchemyPendingReplyAuditRepository(session).create(
            tenant_id=tenant.id,
            pending_reply_id=pending.id,
            action=ValidationAuditAction.ATTACHMENT_ADDED,
            actor_email="admin@example.com",
            detail={"filename": "file.pdf"},
        )
        now = datetime.now(UTC)
        order_repo = SqlAlchemyOrderRepository(session, tenant.id)
        command = OrderCommand(
            action=OrderAction.CREATE,
            name="Client",
            products=(OrderItem(qty=1, product="Widget"),),
        )
        order = order_repo.create_order(
            session_id="email:client@example.com",
            customer_key="client@example.com",
            command=command,
            editable_until=now + timedelta(hours=1),
        )
        order_repo.append_event(
            order_id=order.id,
            session_id="email:client@example.com",
            customer_key="client@example.com",
            action=OrderAction.CREATE,
            result="ok",
            command_json="{}",
            conversation_context=[ChatMessage(role=MessageRole.USER, content="order")],
            created_at=now,
        )
        SqlAlchemyMailDraftRepository(session, tenant_id=tenant.id).create(
            imap_uid="42",
            from_addr="client@example.com",
            to_addr="bot@test.local",
            subject="Hi",
            body_in="Please help",
        )
        SqlAlchemyMailImapUidRepository(session, tenant_id=tenant.id).record_skipped("99")
        TestChatSessionRepository(session, tenant.id).upsert("dashboard:test")
        session.add(
            IngestedFileRow(
                tenant_id=tenant.id,
                path=str(docs_dir / "manual.md"),
                content_hash="abc123",
            )
        )
        session.commit()


def test_flush_removes_operational_data_keeps_rag_and_bot(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    _seed_operational_data(test_settings, tenant)
    slug = tenant.slug

    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        assert _count(session, MessageRow, tenant.id) >= 1
        assert _count(session, HookEventRow, tenant.id) >= 1
        assert _count(session, PendingReplyRow, tenant.id) >= 1
        assert _count(session, PendingReplyEditRow, tenant.id) >= 1
        assert _count(session, PendingReplyAuditEventRow, tenant.id) >= 1
        assert _count(session, OrderRow, tenant.id) >= 1
        assert _count(session, MailDraftRow, tenant.id) >= 1
        assert _count(session, MailImapUidRow, tenant.id) >= 1
        assert _count(session, TestChatSessionRow, tenant.id) >= 1
        assert _count(session, IngestedFileRow, tenant.id) >= 1

        logs = TenantFlushService(session, settings=test_settings).flush(slug)
        session.commit()

    assert any("deleted messages:" in line for line in logs)
    assert any("deleted pending_replies:" in line for line in logs)
    assert any("removed" in line and "attachments" in line for line in logs)
    assert any("removed" in line and "quotes" in line for line in logs)
    assert any("kept ingested_files: 1" in line for line in logs)
    assert any("kept mail_imap_uids: 1" in line for line in logs)

    with factory() as session:
        assert session.get(TenantRow, tenant.id) is not None
        assert _count(session, MessageRow, tenant.id) == 0
        assert _count(session, HookEventRow, tenant.id) == 0
        assert _count(session, PendingReplyRow, tenant.id) == 0
        assert _count(session, PendingReplyEditRow, tenant.id) == 0
        assert _count(session, PendingReplyAuditEventRow, tenant.id) == 0
        assert _count(session, OrderRow, tenant.id) == 0
        assert _count(session, MailDraftRow, tenant.id) == 0
        assert _count(session, TestChatSessionRow, tenant.id) == 0
        assert _count(session, MailImapUidRow, tenant.id) == 1
        assert _count(session, IngestedFileRow, tenant.id) == 1
        connectors = SqlAlchemyConnectorRepository(session).list_for_tenant(tenant.id)
        assert len(connectors) >= 1
        integrations = SqlAlchemyIntegrationRepository(session).list_for_tenant(tenant.id)
        assert len(integrations) >= 1

    assert not (test_settings.data_root / "attachments" / slug).exists()
    assert not (test_settings.data_root / "quotes" / slug).exists()
    assert (test_settings.data_root / "docs" / slug / "manual.md").is_file()


def test_flush_unknown_slug_raises(test_settings, test_tenant) -> None:
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        with pytest.raises(TenantFlushError, match="Unknown tenant slug"):
            TenantFlushService(session, settings=test_settings).flush("no-such-bot")


def test_bot_flush_cli_requires_yes_without_tty(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("chatbot.__main__.get_settings", lambda: test_settings)
        result = runner.invoke(app, ["bot-flush", tenant.slug])
    assert result.exit_code == 1
    assert "--yes" in result.output


def test_bot_flush_cli_with_yes(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    _seed_operational_data(test_settings, tenant)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("chatbot.__main__.get_settings", lambda: test_settings)
        result = runner.invoke(app, ["bot-flush", tenant.slug, "--yes"])
    assert result.exit_code == 0, result.output
    assert "Done." in result.output

    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        assert _count(session, MessageRow, tenant.id) == 0
        assert _count(session, IngestedFileRow, tenant.id) == 1
        assert session.get(TenantRow, tenant.id) is not None
