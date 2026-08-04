from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from evenor.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.validation_regenerate_service import (
    ValidationRegenerateError,
    generate_pending_reply_from_raw,
)
from evenor.config.settings import reset_settings_cache_for_tests
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from evenor.domain.models.fulfillment import FulfillmentKind
from evenor.domain.models.mail_draft import MailDraftStatus
from evenor.domain.models.message import ChatMessage, MessageRole
from evenor.domain.models.tenant import TenantConfig


@pytest.fixture
def regenerate_env(tmp_path, monkeypatch):
    db = tmp_path / "regen.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    monkeypatch.setenv("APP_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("SESSION_SECRET", "sess")
    monkeypatch.setenv("ADMIN_TOKEN", "admin")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    reset_settings_cache_for_tests()
    from evenor.config.settings import get_settings

    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).create(
            slug="regen-bot",
            name="Regen Bot",
            token_hash="x",
            prompt="You are helpful.",
            config=TenantConfig(),
        )
        tenant_id = tenant.id
        from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository

        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        conv = SqlAlchemyConversationRepository(session, tenant_id)
        session_id = "email:client@example.com"
        conv.append_message(
            session_id,
            ChatMessage(role=MessageRole.USER, content="Old sanitized body"),
        )
        conv.append_message(
            session_id,
            ChatMessage(role=MessageRole.ASSISTANT, content="Old draft reply"),
        )
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).create(
            imap_uid="99",
            from_addr="client@example.com",
            to_addr="bot@test.local",
            subject="Hello",
            body_in="<p>Raw HTML inbound</p>",
            body_new="Old sanitized body",
            status=MailDraftStatus.PENDING,
        )
        reply = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id=session_id,
            channel="email",
            recipient_id="client@example.com",
            draft_text="Old draft reply",
            mail_draft_id=draft.id,
        )
        session.commit()
        reply_id = reply.id
    yield factory, settings, tenant_id, reply_id
    engine.dispose()
    reset_settings_cache_for_tests()


def test_generate_uses_raw_body_and_does_not_persist(regenerate_env) -> None:
    factory, settings, tenant_id, reply_id = regenerate_env
    raw_body = "<p>Raw HTML inbound</p>"
    captured: dict = {}

    class _FakeChat:
        def regenerate_assistant_reply(self, session_id, *, history, inbound_text):
            captured["session_id"] = session_id
            captured["history"] = history
            captured["inbound_text"] = inbound_text
            assert inbound_text == raw_body
            assert inbound_text != "Old sanitized body"
            return SimpleNamespace(text="Regenerated reply")

    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        assert tenant is not None
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        assert reply is not None
        result = generate_pending_reply_from_raw(
            session,
            tenant,
            reply,
            settings=settings,
            chat=_FakeChat(),
        )
        assert "Regenerated reply" in result.draft_text
        assert result.draft_html
        assert captured["inbound_text"] == raw_body
        assert reply.draft_text == "Old draft reply"
        conv = SqlAlchemyConversationRepository(session, tenant_id)
        messages = conv.list_messages("email:client@example.com")
        assert messages[-1].content == "Old draft reply"
        assert messages[-2].content == "Old sanitized body"


def test_generate_sends_prior_history_plus_full_raw_body(regenerate_env) -> None:
    factory, settings, tenant_id, reply_id = regenerate_env
    raw_body = "<html><body><p>Full raw with SKU-9999</p></body></html>"
    captured: dict = {}

    with factory() as session:
        from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository

        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        conv = SqlAlchemyConversationRepository(session, tenant_id)
        session_id = "email:thread@example.com"
        conv.append_message(
            session_id, ChatMessage(role=MessageRole.USER, content="First question")
        )
        conv.append_message(
            session_id, ChatMessage(role=MessageRole.ASSISTANT, content="First answer")
        )
        conv.append_message(
            session_id, ChatMessage(role=MessageRole.USER, content="Short cleaned follow-up")
        )
        conv.append_message(
            session_id, ChatMessage(role=MessageRole.ASSISTANT, content="Pending draft")
        )
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).create(
            imap_uid="100",
            from_addr="thread@example.com",
            to_addr="bot@test.local",
            subject="Follow-up",
            body_in=raw_body,
            body_new="Short cleaned follow-up",
        )
        reply = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id=session_id,
            channel="email",
            recipient_id="thread@example.com",
            draft_text="Pending draft",
            mail_draft_id=draft.id,
        )
        session.commit()

        class _FakeChat:
            def regenerate_assistant_reply(self, session_id, *, history, inbound_text):
                captured["history"] = history
                captured["inbound_text"] = inbound_text
                return SimpleNamespace(text="New draft")

        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        assert tenant is not None
        generate_pending_reply_from_raw(
            session,
            tenant,
            reply,
            settings=settings,
            chat=_FakeChat(),
        )

    assert captured["inbound_text"] == raw_body
    assert len(captured["history"]) == 2
    assert captured["history"][0].content == "First question"
    assert captured["history"][1].content == "First answer"


def test_generate_rejects_quote(regenerate_env) -> None:
    factory, settings, tenant_id, reply_id = regenerate_env
    with factory() as session:
        from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository

        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        quote_reply = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:quote@example.com",
            channel="email",
            recipient_id="quote@example.com",
            draft_text="Quote",
            fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
        )
        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        assert tenant is not None
        with pytest.raises(ValidationRegenerateError):
            generate_pending_reply_from_raw(
                session,
                tenant,
                quote_reply,
                settings=settings,
                chat=MagicMock(),
            )
