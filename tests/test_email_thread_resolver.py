from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from evenor.adapters.persistence.email_thread_repository import SqlAlchemyEmailThreadRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.adapters.persistence.outbound_email_message_repository import (
    SqlAlchemyOutboundEmailMessageRepository
)
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.email_thread_disambiguator import EmailThreadDisambiguator
from evenor.application.email_thread_resolver import EmailThreadResolver, InboundEmailHeaders
from evenor.application.tenant_service import TenantService
from evenor.config.settings import get_settings, reset_settings_cache_for_tests


@pytest.fixture
def thread_env(tmp_path, monkeypatch):
    db = tmp_path / "threads.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("EMAIL_THREAD_LLM_ENABLED", "false")
    reset_settings_cache_for_tests()
    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="Thread Bot",
            slug="thread-bot",
        ).tenant
        session.commit()
        tenant_id = tenant.id
    yield factory, settings, tenant_id
    engine.dispose()
    reset_settings_cache_for_tests()


def test_resolver_creates_new_thread(thread_env) -> None:
    factory, settings, tenant_id = thread_env
    with factory() as session:
        resolver = EmailThreadResolver(
            session,
            tenant_id=tenant_id,
            settings=settings,
            disambiguator=EmailThreadDisambiguator(llm=None, enabled=False),
        )
        resolved = resolver.resolve(
            from_addr="client@example.com",
            subject="Devis pompe",
            body_new="Bonjour",
            received_at=datetime(2026, 6, 19, tzinfo=UTC),
            headers=InboundEmailHeaders(),
        )
        assert resolved.created is True
        assert len(resolved.thread_key) == 12
        assert resolved.audit.method == "new_thread"
        assert resolved.audit.used_llm is False


def test_resolver_matches_in_reply_to(thread_env) -> None:
    factory, settings, tenant_id = thread_env
    with factory() as session:
        threads = SqlAlchemyEmailThreadRepository(session, tenant_id=tenant_id)
        thread = threads.create(
            from_addr="client@example.com",
            thread_key="abc123def456",
            root_message_id="<root@example.com>",
            normalized_subject="devis pompe",
            last_activity_at=datetime(2026, 6, 18, tzinfo=UTC),
        )
        drafts = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        drafts.create(
            imap_uid="1",
            from_addr="client@example.com",
            to_addr="bot@example.com",
            subject="Devis pompe",
            body_in="first",
            body_new="first",
            thread_id=thread.id,
            message_id="<parent@example.com>",
        )
        session.commit()

    with factory() as session:
        resolver = EmailThreadResolver(
            session,
            tenant_id=tenant_id,
            settings=settings,
            disambiguator=EmailThreadDisambiguator(llm=None, enabled=False),
        )
        resolved = resolver.resolve(
            from_addr="client@example.com",
            subject="Re: Devis pompe",
            body_new="suite",
            received_at=datetime(2026, 6, 19, tzinfo=UTC),
            headers=InboundEmailHeaders(in_reply_to="<parent@example.com>"),
        )
        assert resolved.thread.id == thread.id
        assert resolved.created is False
        assert resolved.audit.method == "rfc_headers"
        assert resolved.audit.used_llm is False


def test_resolver_matches_normalized_subject(thread_env) -> None:
    factory, settings, tenant_id = thread_env
    with factory() as session:
        threads = SqlAlchemyEmailThreadRepository(session, tenant_id=tenant_id)
        thread = threads.create(
            from_addr="client@example.com",
            thread_key="subjmatch001",
            root_message_id=None,
            normalized_subject="devis pompe",
            last_activity_at=datetime(2026, 6, 18, tzinfo=UTC),
        )
        session.commit()
        thread_id = thread.id

    with factory() as session:
        resolver = EmailThreadResolver(
            session,
            tenant_id=tenant_id,
            settings=settings,
            disambiguator=EmailThreadDisambiguator(llm=None, enabled=False),
        )
        resolved = resolver.resolve(
            from_addr="client@example.com",
            subject="RE: Devis pompe",
            body_new="nouveau",
            received_at=datetime(2026, 6, 19, tzinfo=UTC),
            headers=InboundEmailHeaders(),
        )
        assert resolved.thread.id == thread_id
        assert resolved.audit.method == "subject_exact"


def test_disambiguator_low_confidence_creates_new_thread() -> None:
    llm = MagicMock()
    llm.generate_chat.return_value = MagicMock(
        text='{"same_thread": true, "confidence": 0.2, "thread_key": "abc"}',
        usage=MagicMock(prompt_tokens=10, candidates_tokens=4, total_tokens=14),
    )
    result = EmailThreadDisambiguator(llm=llm, min_confidence=0.7).disambiguate(
        inbound_subject="devis",
        body_preview="hello",
        candidates=[{"thread_key": "abc", "subject": "devis", "last_activity": "2026-01-01"}],
    )
    assert result.same_thread is False
    assert result.llm_called is True
    assert result.prompt_tokens == 10


def test_resolver_does_not_call_llm_on_rfc_match(thread_env) -> None:
    factory, settings, tenant_id = thread_env
    llm = MagicMock()
    with factory() as session:
        threads = SqlAlchemyEmailThreadRepository(session, tenant_id=tenant_id)
        thread = threads.create(
            from_addr="client@example.com",
            thread_key="abc123def456",
            root_message_id="<root@example.com>",
            normalized_subject="devis pompe",
            last_activity_at=datetime(2026, 6, 18, tzinfo=UTC),
        )
        drafts = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        drafts.create(
            imap_uid="1",
            from_addr="client@example.com",
            to_addr="bot@example.com",
            subject="Devis pompe",
            body_in="first",
            body_new="first",
            thread_id=thread.id,
            message_id="<parent@example.com>",
        )
        session.commit()

    with factory() as session:
        resolver = EmailThreadResolver(
            session,
            tenant_id=tenant_id,
            settings=settings,
            disambiguator=EmailThreadDisambiguator(llm=llm, enabled=True),
        )
        resolver.resolve(
            from_addr="client@example.com",
            subject="Re: Devis pompe",
            body_new="suite",
            received_at=datetime(2026, 6, 19, tzinfo=UTC),
            headers=InboundEmailHeaders(in_reply_to="<parent@example.com>"),
        )
    llm.generate_chat.assert_not_called()
