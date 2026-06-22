from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chatbot.adapters.mail.imap_client import IncomingMail
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.mail_imap_uid_repository import SqlAlchemyMailImapUidRepository
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.config.settings import get_settings, reset_settings_cache_for_tests
from chatbot.application.tenant_service import TenantService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.mail import listener as mail_listener
from chatbot.mail.process_since import parse_process_since


@pytest.fixture
def mail_env(tmp_path, monkeypatch):
    db = tmp_path / "mail.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    from cryptography.fernet import Fernet

    monkeypatch.setenv("APP_SECRET_KEY", Fernet.generate_key().decode())
    reset_settings_cache_for_tests()
    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="Mail Bot",
            slug="mail-bot",
        ).tenant
        conn_repo = SqlAlchemyConnectorRepository(session)
        in_conn = conn_repo.create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "imap_host": "greenmail",
                "imap_port": "3143",
                "username": "bot@test.local",
                "password": "secret",
            },
        )
        out_conn = conn_repo.create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "outbound_provider": "smtp",
                "smtp_host": "mailpit",
                "smtp_port": "1025",
                "smtp_use_tls": "false",
                "from_addr": "bot@test.local",
            },
        )
        session.commit()
        tenant_id = tenant.id
        in_id = in_conn.id
        out_id = out_conn.id
    yield factory, settings, tenant_id, in_id, out_id
    engine.dispose()
    reset_settings_cache_for_tests()


def _mail(
    uid: str,
    from_addr: str = "client@example.com",
    *,
    to_addr: str = "bot@test.local",
    to_addrs: tuple[str, ...] | None = None,
    cc_addrs: tuple[str, ...] = (),
) -> IncomingMail:
    primary = to_addrs if to_addrs is not None else ((to_addr,) if to_addr else ())
    return IncomingMail(
        uid=uid,
        from_addr=from_addr,
        to_addr=to_addr,
        subject="Test",
        body_text="Hello",
        to_addrs=primary,
        cc_addrs=cc_addrs,
    )


@patch("chatbot.mail.listener.imap_client")
def test_run_once_blacklisted_sender_skips_llm(mock_imap_ctx, mail_env) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    with factory() as session:
        from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
        from chatbot.application.tenant_service import TenantService

        TenantService(SqlAlchemyTenantRepository(session)).update_blocked_senders(
            tenant_id, ["blocked@evil.com"]
        )
        session.commit()

    imap = MagicMock()
    imap.fetch_pending.return_value = [_mail("blk-1", from_addr="blocked@evil.com")]
    mock_imap_ctx.return_value.__enter__.return_value = imap

    with patch("chatbot.mail.listener.build_chat_service_for_worker") as mock_build:
        n = mail_listener.run_once(factory, settings)
        mock_build.assert_not_called()

    assert n == 0
    with factory() as session:
        from chatbot.adapters.persistence.mail_imap_uid_repository import (
            SqlAlchemyMailImapUidRepository,
        )

        uid_repo = SqlAlchemyMailImapUidRepository(session, tenant_id=tenant_id)
        assert uid_repo.exists_by_uid("blk-1")


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_processes_mail(mock_imap_ctx, mock_build_chat, mock_queue, mail_env) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    imap = MagicMock()
    imap.fetch_pending.return_value = [_mail("1")]
    mock_imap_ctx.return_value.__enter__.return_value = imap
    mock_build_chat.return_value.handle_user_message.return_value = SimpleNamespace(
        text="reply",
        hook_type=None,
        hook_payload_json=None,
        hook_event_id=None,
    )
    mock_queue.return_value = ("queued", None)

    n = mail_listener.run_once(factory, settings)
    assert n == 1
    imap.mark_seen.assert_not_called()
    mock_queue.assert_called_once()


@patch("chatbot.mail.listener.imap_client")
def test_run_once_tenant_failure_does_not_block_other(mock_imap_ctx, mail_env) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    with factory() as session:
        tenant_b = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="Mail Bot 2",
            slug="mail-bot-2",
        ).tenant
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_b.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "imap_host": "greenmail",
                "imap_port": "3143",
                "username": "bot2@test.local",
                "password": "secret",
            },
        )
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_b.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={
                "outbound_provider": "smtp",
                "smtp_host": "mailpit",
                "smtp_port": "1025",
                "from_addr": "bot2@test.local",
            },
        )
        session.commit()

    call_count = {"n": 0}

    def _side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("IMAP down")
        imap = MagicMock()
        imap.fetch_pending.return_value = []
        return MagicMock(__enter__=MagicMock(return_value=imap), __exit__=MagicMock(return_value=False))

    mock_imap_ctx.side_effect = _side_effect
    n = mail_listener.run_once(factory, settings)
    assert n == 0
    assert call_count["n"] == 2


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_mail_failure_continues(mock_imap_ctx, mock_build_chat, mock_queue, mail_env) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    imap = MagicMock()
    imap.fetch_pending.return_value = [_mail("1"), _mail("2", "other@example.com")]
    mock_imap_ctx.return_value.__enter__.return_value = imap

    def _chat_side_effect(session, settings, tenant):
        svc = MagicMock()

        def _handle(session_id, body):
            if "other@" in session_id:
                raise RuntimeError("LLM fail")
            return SimpleNamespace(text="ok", hook_type=None, hook_payload_json=None, hook_event_id=None)

        svc.handle_user_message.side_effect = _handle
        return svc

    mock_build_chat.side_effect = _chat_side_effect
    mock_queue.return_value = ("queued", None)
    n = mail_listener.run_once(factory, settings)
    assert n == 1
    imap.mark_seen.assert_not_called()


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_skips_mail_before_process_since(
    mock_imap_ctx, mock_build_chat, mock_queue, mail_env
) -> None:
    factory, settings, tenant_id, in_id, _ = mail_env
    with factory() as session:
        repo = SqlAlchemyConnectorRepository(session)
        existing = repo.find_by_id(in_id)
        assert existing is not None
        cfg = dict(existing.config)
        cfg["process_since"] = "2026-06-08T12:00:00+00:00"
        repo.update(in_id, config=cfg)
        session.commit()

    imap = MagicMock()
    imap.fetch_pending.return_value = [
        IncomingMail(
            uid="old-1",
            from_addr="client@example.com",
            to_addr="bot@test.local",
            subject="Old",
            body_text="Old body",
            received_at=datetime(2026, 6, 7, 10, 0, tzinfo=UTC),
        )
    ]
    mock_imap_ctx.return_value.__enter__.return_value = imap

    n = mail_listener.run_once(factory, settings)
    assert n == 0
    mock_build_chat.assert_not_called()
    mock_queue.assert_not_called()
    imap.mark_seen.assert_not_called()

    with factory() as session:
        uid_repo = SqlAlchemyMailImapUidRepository(session, tenant_id=tenant_id)
        draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        assert uid_repo.exists_by_uid("old-1")
        assert not draft_repo.exists_by_uid("old-1")


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_direct_mode_marks_seen(mock_imap_ctx, mock_build_chat, mock_queue, mail_env) -> None:
    factory, settings, tenant_id, _, out_id = mail_env
    with factory() as session:
        repo = SqlAlchemyConnectorRepository(session)
        out = repo.find_by_id(out_id)
        assert out is not None
        repo.update(out_id, mode=ConnectorMode.DIRECT)
        session.commit()

    imap = MagicMock()
    imap.fetch_pending.return_value = [_mail("1")]
    mock_imap_ctx.return_value.__enter__.return_value = imap
    mock_build_chat.return_value.handle_user_message.return_value = SimpleNamespace(
        text="reply",
        hook_type=None,
        hook_payload_json=None,
        hook_event_id=None,
    )
    mock_queue.return_value = ("ok", None)

    n = mail_listener.run_once(factory, settings)
    assert n == 1
    imap.mark_seen.assert_called_once_with("1")


@patch("chatbot.mail.listener.imap_client")
def test_run_once_persists_process_since_for_existing_connector(mock_imap_ctx, mail_env) -> None:
    factory, settings, tenant_id, in_id, _ = mail_env
    imap = MagicMock()
    imap.fetch_pending.return_value = []
    mock_imap_ctx.return_value.__enter__.return_value = imap

    mail_listener.run_once(factory, settings)

    with factory() as session:
        conn = ConnectorService(SqlAlchemyConnectorRepository(session)).get(in_id)
        assert conn is not None
        assert parse_process_since(conn.config) is not None


def test_is_cc_only_defaults_to_enabled() -> None:
    mail = _mail("1", to_addr="client@example.com", cc_addrs=("bot@test.local",))
    assert mail_listener._is_cc_only(mail, {}, "bot@test.local") is True


def test_is_cc_only_disabled_when_config_off() -> None:
    mail = _mail("1", to_addr="client@example.com", cc_addrs=("bot@test.local",))
    assert mail_listener._is_cc_only(mail, {"skip_cc_only": False}, "bot@test.local") is False


@patch("chatbot.mail.listener.imap_client")
def test_run_once_skips_cc_only_mail_by_default(mock_imap_ctx, mail_env) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    imap = MagicMock()
    imap.fetch_pending.return_value = [
        _mail(
            "cc-1",
            to_addr="client@example.com",
            to_addrs=("client@example.com",),
            cc_addrs=("bot@test.local",),
        )
    ]
    mock_imap_ctx.return_value.__enter__.return_value = imap

    with patch("chatbot.mail.listener.build_chat_service_for_worker") as mock_build:
        n = mail_listener.run_once(factory, settings)
        mock_build.assert_not_called()

    assert n == 0
    with factory() as session:
        uid_repo = SqlAlchemyMailImapUidRepository(session, tenant_id=tenant_id)
        draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        assert uid_repo.exists_by_uid("cc-1")
        assert not draft_repo.exists_by_uid("cc-1")


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_processes_mail_when_mailbox_in_to(
    mock_imap_ctx, mock_build_chat, mock_queue, mail_env
) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    imap = MagicMock()
    imap.fetch_pending.return_value = [
        _mail(
            "to-1",
            to_addr="bot@test.local",
            to_addrs=("bot@test.local", "client@example.com"),
        )
    ]
    mock_imap_ctx.return_value.__enter__.return_value = imap
    mock_build_chat.return_value.handle_user_message.return_value = SimpleNamespace(
        text="reply",
        hook_type=None,
        hook_payload_json=None,
        hook_event_id=None,
    )
    mock_queue.return_value = ("queued", None)

    n = mail_listener.run_once(factory, settings)
    assert n == 1


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_processes_cc_only_when_filter_disabled(
    mock_imap_ctx, mock_build_chat, mock_queue, mail_env
) -> None:
    factory, settings, tenant_id, in_id, _ = mail_env
    with factory() as session:
        repo = SqlAlchemyConnectorRepository(session)
        existing = repo.find_by_id(in_id)
        assert existing is not None
        cfg = dict(existing.config)
        cfg["skip_cc_only"] = False
        repo.update(in_id, config=cfg)
        session.commit()

    imap = MagicMock()
    imap.fetch_pending.return_value = [
        _mail(
            "cc-2",
            to_addr="client@example.com",
            to_addrs=("client@example.com",),
            cc_addrs=("bot@test.local",),
        )
    ]
    mock_imap_ctx.return_value.__enter__.return_value = imap
    mock_build_chat.return_value.handle_user_message.return_value = SimpleNamespace(
        text="reply",
        hook_type=None,
        hook_payload_json=None,
        hook_event_id=None,
    )
    mock_queue.return_value = ("queued", None)

    n = mail_listener.run_once(factory, settings)
    assert n == 1


@patch("chatbot.mail.listener.queue_after_chat")
@patch("chatbot.mail.listener.build_chat_service_for_worker")
@patch("chatbot.mail.listener.imap_client")
def test_run_once_processes_mail_when_mailbox_in_to_and_cc(
    mock_imap_ctx, mock_build_chat, mock_queue, mail_env
) -> None:
    factory, settings, tenant_id, _, _ = mail_env
    imap = MagicMock()
    imap.fetch_pending.return_value = [
        _mail(
            "both-1",
            to_addr="bot@test.local",
            to_addrs=("bot@test.local",),
            cc_addrs=("bot@test.local", "client@example.com"),
        )
    ]
    mock_imap_ctx.return_value.__enter__.return_value = imap
    mock_build_chat.return_value.handle_user_message.return_value = SimpleNamespace(
        text="reply",
        hook_type=None,
        hook_payload_json=None,
        hook_event_id=None,
    )
    mock_queue.return_value = ("queued", None)

    n = mail_listener.run_once(factory, settings)
    assert n == 1
