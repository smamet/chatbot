from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.validation_translate_service import (
    ValidationTranslateError,
    translate_pending_reply_draft,
)
from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.tenant import TenantConfig


@pytest.fixture
def translate_env(tmp_path, monkeypatch):
    db = tmp_path / "translate.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    monkeypatch.setenv("APP_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("SESSION_SECRET", "sess")
    monkeypatch.setenv("ADMIN_TOKEN", "admin")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    reset_settings_cache_for_tests()
    from chatbot.config.settings import get_settings

    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).create(
            slug="translate-bot",
            name="Translate Bot",
            token_hash="x",
            prompt="You are helpful.",
            config=TenantConfig(),
        )
        tenant_id = tenant.id
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        reply = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Bonjour",
            draft_html="<p>Bonjour</p>",
        )
        session.commit()
        reply_id = reply.id
    yield factory, settings, tenant_id, reply_id
    engine.dispose()
    reset_settings_cache_for_tests()


def test_translate_calls_llm_and_sanitizes_html(translate_env) -> None:
    factory, settings, tenant_id, reply_id = translate_env
    captured: dict = {}

    class _FakeLlm:
        def generate_chat(self, *, system_instruction, messages):
            captured["system"] = system_instruction
            captured["user"] = messages[0].content
            return SimpleNamespace(
                text='{"draft_html": "<p>Hello</p>", "draft_subject": "Re: Test"}'
            )

    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        assert tenant is not None and reply is not None
        out = translate_pending_reply_draft(
            reply,
            draft_html="<p>Bonjour</p>",
            draft_subject="Objet",
            target_lang="en",
            tenant=tenant,
            settings=settings,
            session=session,
            llm=_FakeLlm(),
        )

    assert out["draft_html"] == "<p>Hello</p>"
    assert out["draft_subject"] == "Re: Test"
    assert "English" in captured["system"]
    assert "Target language: en" in captured["user"]
    assert "Objet" in captured["user"]


def test_translate_rejects_invalid_target_lang(translate_env) -> None:
    factory, settings, tenant_id, reply_id = translate_env
    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        assert tenant is not None and reply is not None
        with pytest.raises(ValidationTranslateError, match="target_lang"):
            translate_pending_reply_draft(
                reply,
                draft_html="<p>Hi</p>",
                draft_subject="",
                target_lang="de",
                tenant=tenant,
                settings=settings,
                session=session,
                llm=MagicMock(),
            )


def test_translate_rejects_non_email(translate_env) -> None:
    factory, settings, tenant_id, reply_id = translate_env
    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.VALIDATION,
            config={},
        )
        reply = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="wa:123",
            channel="whatsapp",
            recipient_id="123",
            draft_text="Hi",
            draft_html="<p>Hi</p>",
        )
        assert tenant is not None
        with pytest.raises(ValidationTranslateError, match="email"):
            translate_pending_reply_draft(
                reply,
                draft_html="<p>Hi</p>",
                draft_subject="",
                target_lang="fr",
                tenant=tenant,
                settings=settings,
                session=session,
                llm=MagicMock(),
            )
