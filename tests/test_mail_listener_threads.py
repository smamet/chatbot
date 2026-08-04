from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from evenor.adapters.mail.imap_client import IncomingMail
from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.tenant_service import TenantService
from evenor.config.settings import get_settings, reset_settings_cache_for_tests
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from evenor.mail import listener as mail_listener


@pytest.fixture
def listener_env(tmp_path, monkeypatch):
    db = tmp_path / "listener.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("EMAIL_THREAD_LLM_ENABLED", "false")
    from cryptography.fernet import Fernet

    monkeypatch.setenv("APP_SECRET_KEY", Fernet.generate_key().decode())
    reset_settings_cache_for_tests()
    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="Listener Bot",
            slug="listener-bot",
        ).tenant
        conn_repo = SqlAlchemyConnectorRepository(session)
        conn_repo.create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"imap_host": "greenmail", "imap_port": "3143", "username": "u", "password": "p"},
        )
        conn_repo.create(
            tenant_id=tenant.id,
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
        session.commit()
        tenant_id = tenant.id
    yield factory, settings, tenant_id
    engine.dispose()
    reset_settings_cache_for_tests()


@patch("evenor.mail.listener.queue_after_chat")
@patch("evenor.mail.listener.build_chat_service_for_worker")
@patch("evenor.mail.listener.imap_client")
def test_listener_uses_thread_session_and_body_new(
    mock_imap_ctx, mock_build_chat, mock_queue, listener_env
) -> None:
    factory, settings, tenant_id = listener_env
    imap = MagicMock()
    imap.fetch_pending.return_value = [
        IncomingMail(
            uid="42",
            from_addr="client@example.com",
            to_addr="bot@test.local",
            subject="Devis pompe",
            body_text="Ma question\n\nOn Mon wrote:\n> old",
            message_id="<in@example.com>",
        )
    ]
    mock_imap_ctx.return_value.__enter__.return_value = imap
    chat_svc = MagicMock()
    mock_build_chat.return_value = chat_svc
    chat_svc.handle_user_message.return_value = SimpleNamespace(
        text="reply",
        hook_type=None,
        hook_payload_json=None,
        hook_event_id=None,
    )
    mock_queue.return_value = ("queued", None)

    n = mail_listener.run_once(factory, settings)
    assert n == 1
    session_id, body = chat_svc.handle_user_message.call_args[0]
    assert session_id.startswith("email:client@example.com~")
    assert body == "Ma question"
    assert mock_queue.call_args.kwargs["mail_draft_id"] is not None
    assert mock_queue.call_args.kwargs["thread_id"] is not None

    with factory() as session:
        draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        assert draft_repo.exists_by_uid("42")
        draft = draft_repo.find_by_message_id("<in@example.com>")
        assert draft is not None
        assert draft.body_new == "Ma question"
        assert draft.thread_resolution_json is not None
        assert '"method"' in draft.thread_resolution_json
