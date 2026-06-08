from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
from chatbot.application.integration_service import IntegrationService
from chatbot.application.tenant_service import TenantService
from chatbot.application.user_service import UserService
from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.integration import IntegrationType
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.domain.models.user import UserRole
from chatbot.interfaces.api.deps import _build_chat_service
from chatbot.interfaces.api.main import create_app, refresh_genai_clients_if_needed


@pytest.fixture
def dashboard_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    data = tmp_path / "data"
    db = data / "dash.db"
    secret = Fernet.generate_key().decode()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LANCEDB_ROOT", str(data / "lancedb"))
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    monkeypatch.setenv("APP_SECRET_KEY", secret)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    reset_settings_cache_for_tests()
    from chatbot.config.settings import get_settings

    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        user_svc = UserService(SqlAlchemyUserRepository(session))
        admin = user_svc.create_user(
            email="admin@test.com", password="admin-pass", role=UserRole.ADMIN
        )
        editor = user_svc.create_user(
            email="editor@test.com", password="edit-pass", role=UserRole.CLIENT_ADMIN
        )
        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        result = tenant_svc.create_tenant(name="Dash Bot", slug="dash-bot")
        user_svc.grant_access(editor.id, result.tenant.id)
        session.commit()
        slug = result.tenant.slug
        tenant_id = result.tenant.id
    app = create_app()
    app.state.session_factory = factory
    refresh_genai_clients_if_needed(app)

    class _FakeChat:
        def handle_user_message(self, session_id: str, message: str, *, attachments=None):
            _ = attachments
            return SimpleNamespace(
                text=f"echo:{message}",
                usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            )

    app.dependency_overrides.clear()

    original_build = _build_chat_service

    def _patched_build(request, settings, tenant, repo, hook_repo, **kwargs):
        _ = request, settings, tenant, hook_repo, kwargs
        fake = _FakeChat()

        class _Wrapper:
            def handle_user_message(self, session_id: str, message: str, *, attachments=None):
                out = fake.handle_user_message(session_id, message, attachments=attachments)
                repo.append_message(session_id, ChatMessage(role=MessageRole.USER, content=message))
                repo.append_message(
                    session_id, ChatMessage(role=MessageRole.ASSISTANT, content=out.text)
                )
                return out

        return _Wrapper()

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    dash_mod._build_chat_service = _patched_build  # type: ignore[attr-defined]

    with TestClient(app) as client:
        yield client, admin, editor, slug, tenant_id, data, factory
    dash_mod._build_chat_service = original_build  # type: ignore[attr-defined]
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()
    engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post("/auth/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303


def test_dashboard_requires_login(dashboard_env) -> None:
    client, *_ = dashboard_env
    assert client.get("/dashboard/bots").status_code == 401


def test_admin_bot_crud(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")

    r = client.get(f"/dashboard/bots/{slug}")
    assert r.status_code == 200
    assert "Dash Bot" in r.text

    r = client.post(
        f"/dashboard/bots/{slug}/settings",
        data={"name": "Renamed Bot", "active": "on"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    r = client.post(
        f"/dashboard/bots/{slug}/rag-config",
        data={
            "chat_model": "gemini-2.5-flash",
            "embedding_model": "gemini-embedding-001",
            "rewrite_model": "gemini-2.5-flash",
            "rag_enabled": "on",
            "rag_rewrite_enabled": "on",
            "rag_top_k": "3",
            "chunk_size": "400",
            "chunk_overlap": "50",
            "retrieval_language": "fr",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with factory() as session:
        from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository

        saved = SqlAlchemyTenantRepository(session).find_by_slug(slug)
        assert saved is not None
        assert saved.config.rag_enabled is True
        assert saved.config.rag_rewrite_enabled is True
        assert saved.config.rag_top_k == 3
        assert saved.config.retrieval_language == "fr"

    docs_dir = data / "docs" / slug
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "note.md").write_text("hello", encoding="utf-8")

    class _FakeEmbedder:
        dim = 4

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.dim for _ in texts]

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    old_embedder = dash_mod.GeminiEmbedder
    dash_mod.GeminiEmbedder = _FakeEmbedder  # type: ignore[misc, assignment]
    try:
        r = client.post(
            f"/dashboard/bots/{slug}/sync",
            data={"fresh": "on"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        r = client.get(f"/dashboard/bots/{slug}?tab=documents")
        assert "note.md" in r.text
    finally:
        dash_mod.GeminiEmbedder = old_embedder  # type: ignore[misc]

    with factory() as session:
        conn_svc_repo = SqlAlchemyConnectorRepository(session)
        from chatbot.application.connector_service import ConnectorService

        svc = ConnectorService(conn_svc_repo)
        conn = svc.upsert(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.DIRECT,
            config={"verify_token": "vt"},
        )
        session.commit()
        conn_id = conn.id

    r = client.post(
        f"/dashboard/bots/{slug}/connectors/{conn_id}/toggle",
        follow_redirects=False,
    )
    assert r.status_code == 303
    r = client.get(f"/dashboard/bots/{slug}?tab=connectors")
    assert "inactive" in r.text
    with factory() as session:
        from chatbot.application.connector_service import ConnectorService

        conn = ConnectorService(SqlAlchemyConnectorRepository(session)).get(conn_id)
        assert conn is not None
        assert conn.active is False

    r = client.post(
        f"/dashboard/bots/{slug}/connectors/{conn_id}/toggle",
        follow_redirects=False,
    )
    assert r.status_code == 303
    r = client.get(f"/dashboard/bots/{slug}?tab=connectors")
    assert 'badge on">active' in r.text or "badge on\">active" in r.text

    r = client.post(
        f"/dashboard/bots/{slug}/chat-test/send",
        data={"message": "hello dash"},
    )
    assert r.status_code == 200
    assert r.json()["reply"] == "echo:hello dash"
    r = client.get(f"/dashboard/bots/{slug}?tab=chat")
    assert "echo:hello dash" in r.text
    assert "chat-test.js" in r.text
    assert "markdown.js" in r.text

    r = client.post(
        f"/dashboard/bots/{slug}/chat-test/reset",
        follow_redirects=False,
    )
    assert r.status_code == 303
    r = client.get(f"/dashboard/bots/{slug}?tab=chat")
    assert "echo:hello dash" not in r.text

    with factory() as session:
        from chatbot.adapters.persistence.conversation_repository import (
            SqlAlchemyConversationRepository,
        )

        repo = SqlAlchemyConversationRepository(session, tenant_id)
        repo.append_message(
            "email:foo@bar.com",
            ChatMessage(role=MessageRole.USER, content="hello"),
        )
        repo.append_message(
            "email:foo@bar.com",
            ChatMessage(role=MessageRole.ASSISTANT, content="**bold** reply"),
        )
        session.commit()

    r = client.get(f"/dashboard/bots/{slug}?tab=history&sid=email%3Afoo%40bar.com")
    assert r.status_code == 200
    assert "foo@bar.com" in r.text
    assert "email:foo@bar.com" not in r.text
    assert 'class="msg-body js-md"' in r.text
    assert "**bold** reply" in r.text
    assert "markdown.js" in r.text


def test_editor_cannot_delete_bot(dashboard_env) -> None:
    client, _, editor, slug, *_ = dashboard_env
    _login(client, editor.email, "edit-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/delete",
        data={"confirm": slug},
        follow_redirects=False,
    )
    assert r.status_code in (401, 403)


def test_user_management(dashboard_env) -> None:
    client, admin, editor, slug, tenant_id, *_ = dashboard_env
    _login(client, admin.email, "admin-pass")

    r = client.get("/dashboard/users")
    assert r.status_code == 200
    assert editor.email in r.text

    r = client.get(f"/dashboard/users/{editor.id}")
    assert r.status_code == 200

    r = client.post(
        f"/dashboard/users/{editor.id}/role",
        data={"role": UserRole.CLIENT_USER.value},
        follow_redirects=False,
    )
    assert r.status_code == 303

    r = client.post(
        f"/dashboard/users/{editor.id}/access",
        data={f"tenant_{tenant_id}": "on"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    _login(client, editor.email, "edit-pass")
    r = client.get(f"/dashboard/bots/{slug}")
    assert r.status_code == 200


def test_admin_delete_bot(dashboard_env) -> None:
    client, admin, _, slug, *_ = dashboard_env
    _login(client, admin.email, "admin-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/delete",
        data={"confirm": slug},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert client.get(f"/dashboard/bots/{slug}").status_code == 404


def test_bot_export_import(dashboard_env) -> None:
    client, admin, editor, slug, tenant_id, data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")

    docs_dir = data / "docs" / slug
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "sample.md").write_text("sample doc", encoding="utf-8")

    with factory() as session:
        from chatbot.application.connector_service import ConnectorService

        ConnectorService(SqlAlchemyConnectorRepository(session)).upsert(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.DIRECT,
            config={"verify_token": "export-vt"},
        )
        from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository

        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        tenant_svc.update_tenant(
            tenant_id,
            prompt="Export prompt",
            hook_instructions="Export hooks",
            gemini_api_key="export-gemini",
        )
        session.commit()

    r = client.get(f"/dashboard/bots/{slug}/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    zip_bytes = r.content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["source_slug"] == slug
        assert manifest["bot"]["prompt"] == "Export prompt"
        assert "documents/sample.md" in zf.namelist()

    r = client.get("/dashboard/bots/import")
    assert r.status_code == 200
    assert "Import bot" in r.text

    r = client.post(
        "/dashboard/bots/import",
        data={"mode": "create", "new_name": "Imported Bot"},
        files={"bundle": ("bot.zip", zip_bytes, "application/zip")},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert "Imported Bot" in r.text

    with factory() as session:
        tenants = TenantService(SqlAlchemyTenantRepository(session)).list_tenants()
    imported = next(t for t in tenants if t.name == "Imported Bot")
    imported_docs = data / "docs" / imported.slug
    assert (imported_docs / "sample.md").read_text(encoding="utf-8") == "sample doc"

    r = client.post(
        "/dashboard/bots/import",
        data={"mode": "overwrite", "target_slug": slug, "new_name": "Overwritten"},
        files={"bundle": ("bot.zip", zip_bytes, "application/zip")},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/dashboard/bots/{slug}?tab=config"

    with factory() as session:
        saved = SqlAlchemyTenantRepository(session).find_by_slug(slug)
        assert saved is not None
        assert saved.name == "Overwritten"
        assert saved.prompt == "Export prompt"


def test_client_user_cannot_export_import(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    with factory() as session:
        reader = UserService(SqlAlchemyUserRepository(session)).create_user(
            email="reader@test.com",
            password="read-pass",
            role=UserRole.CLIENT_USER,
        )
        UserService(SqlAlchemyUserRepository(session)).grant_access(reader.id, tenant_id)
        session.commit()

    _login(client, reader.email, "read-pass")
    assert client.get(f"/dashboard/bots/{slug}/export").status_code == 403
    assert client.get("/dashboard/bots/import").status_code == 403

    docs_dir = data / "docs" / slug
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "x.md").write_text("x", encoding="utf-8")
    _login(client, admin.email, "admin-pass")
    zip_bytes = client.get(f"/dashboard/bots/{slug}/export").content
    _login(client, reader.email, "read-pass")
    r = client.post(
        "/dashboard/bots/import",
        data={"mode": "create", "new_name": "Nope"},
        files={"bundle": ("bot.zip", zip_bytes, "application/zip")},
    )
    assert r.status_code == 403


def test_integration_dashboard_save(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")

    r = client.get(f"/dashboard/bots/{slug}?tab=integrations")
    assert r.status_code == 200
    assert "ERPNext" in r.text
    assert "Integrations" in r.text

    r = client.post(
        f"/dashboard/bots/{slug}/integrations",
        data={
            "integration_type": "erpnext",
            "url": "https://erp.example.com",
            "api_key": "test-key",
            "api_secret": "test-secret",
            "identity_email_field": "email_id",
            "identity_phone_field": "mobile_no",
            "fetch_orders": "on",
            "fetch_quotations": "on",
            "max_items": "5",
            "active": "on",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/dashboard/bots/{slug}?tab=integrations&integration_type=erpnext"

    with factory() as session:
        integration = IntegrationService(
            SqlAlchemyIntegrationRepository(session)
        ).find_active(tenant_id, type=IntegrationType.ERPNEXT)
        assert integration is not None
        assert integration.config["url"] == "https://erp.example.com"
        assert integration.config["fetch_orders"] is True


def test_integration_test_endpoint(dashboard_env) -> None:
    client, admin, _, slug, _tenant_id, _data, _factory = dashboard_env
    _login(client, admin.email, "admin-pass")

    r = client.post(
        f"/dashboard/bots/{slug}/integrations/test",
        data={
            "integration_type": "erpnext",
            "url": "https://erp.example.com",
            "api_key": "k",
            "api_secret": "s",
            "identity_email_field": "email_id",
            "identity_phone_field": "mobile_no",
            "max_items": "5",
            "test_email": "alice@example.com",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body


def test_quickbooks_connect_requires_saved_config(dashboard_env) -> None:
    client, admin, _, slug, _tenant_id, _data, _factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    r = client.get(
        f"/dashboard/bots/{slug}/integrations/quickbooks/connect",
        follow_redirects=False,
    )
    assert r.status_code == 400


def _upsert_email_connectors(client: TestClient, slug: str) -> None:
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "in",
            "mode": "validation",
            "active": "on",
            "imap_host": "greenmail",
            "imap_port": "3143",
            "username": "bot@test.local",
            "password": "secret",
        },
        follow_redirects=False,
    )
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "out",
            "mode": "validation",
            "active": "on",
            "outbound_provider": "smtp",
            "from_addr": "bot@test.local",
            "smtp_host": "mailpit",
            "smtp_port": "1025",
            "smtp_use_tls": "",
        },
        follow_redirects=False,
    )


def test_email_test_tab_hidden_without_dev_mode(dashboard_env, monkeypatch) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    monkeypatch.setenv("DEV_MODE", "false")
    reset_settings_cache_for_tests()
    with factory() as session:
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"imap_host": "greenmail", "username": "bot@test.local", "password": "s"},
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.get(f"/dashboard/bots/{slug}")
    assert r.status_code == 200
    assert "Test email" not in r.text


def test_email_test_tab_visible_in_dev_mode(dashboard_env, monkeypatch) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    monkeypatch.setenv("DEV_MODE", "true")
    reset_settings_cache_for_tests()
    with factory() as session:
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"imap_host": "greenmail", "username": "bot@test.local", "password": "s"},
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.get(f"/dashboard/bots/{slug}?tab=email-test")
    assert r.status_code == 200
    assert "Test email" in r.text
    assert "email-test.js" in r.text
    assert "Mailpit" in r.text


def test_email_test_send_forbidden_without_dev_mode(dashboard_env, monkeypatch) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    monkeypatch.setenv("DEV_MODE", "false")
    reset_settings_cache_for_tests()
    with factory() as session:
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"imap_host": "greenmail", "username": "bot@test.local", "password": "s"},
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/email-test/send",
        data={"from_addr": "client@example.com", "subject": "Hi", "body": "Test"},
    )
    assert r.status_code == 403


@patch("chatbot.interfaces.api.routers.dashboard_web.inject_test_email")
def test_email_test_send_ok(mock_inject, dashboard_env, monkeypatch) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    monkeypatch.setenv("DEV_MODE", "true")
    reset_settings_cache_for_tests()
    with factory() as session:
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"imap_host": "greenmail", "username": "bot@test.local", "password": "s"},
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/email-test/send",
        data={"from_addr": "client@example.com", "subject": "Hi", "body": "Test body"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    mock_inject.assert_called_once()
    call_args = mock_inject.call_args
    assert call_args[0][1]["username"] == "bot@test.local"


@patch("chatbot.interfaces.api.routers.dashboard_web.poll_tenant_now")
def test_email_test_poll_ok(mock_poll, dashboard_env, monkeypatch) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    monkeypatch.setenv("DEV_MODE", "true")
    reset_settings_cache_for_tests()
    mock_poll.return_value = 2
    with factory() as session:
        SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"imap_host": "greenmail", "username": "bot@test.local", "password": "s"},
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.post(f"/dashboard/bots/{slug}/email-test/poll")
    assert r.status_code == 200
    body = r.json()
    assert body["processed_mails"] == 2


def test_validation_tab_renders_markdown_and_session_label(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
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
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="**Hello** client",
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.get(f"/dashboard/bots/{slug}?tab=validation")
    assert r.status_code == 200
    assert "client@example.com" in r.text
    assert "email:client@example.com" not in r.text
    assert 'class="validation-message-body msg-body js-md"' in r.text
    assert "**Hello** client" in r.text
    assert "markdown.js" in r.text


def test_delete_connector_removes_pending_replies(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
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
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Queued reply",
        )
        session.commit()
        conn_id = connector.id

    _login(client, admin.email, "admin-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/connectors/{conn_id}/delete",
        follow_redirects=False,
    )
    assert r.status_code == 303

    with factory() as session:
        from chatbot.application.connector_service import ConnectorService

        assert ConnectorService(SqlAlchemyConnectorRepository(session)).get(conn_id) is None
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        assert SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id) == []


def test_email_in_connector_sets_process_since_on_save(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "in",
            "mode": "direct",
            "active": "on",
            "imap_host": "imap.example.com",
            "imap_port": "993",
            "username": "bot@example.com",
            "password": "secret",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with factory() as session:
        from chatbot.application.connector_service import ConnectorService
        from chatbot.mail.process_since import parse_process_since

        conn = ConnectorService(SqlAlchemyConnectorRepository(session)).find(
            tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
        )
        assert conn is not None
        assert parse_process_since(conn.config) is not None

    r = client.get(f"/dashboard/bots/{slug}?tab=connectors")
    assert "connector-configs-data" in r.text
    assert "email:in" in r.text
    assert "process_since" in r.text
    assert "Deactivate" in r.text or "Activate" in r.text
