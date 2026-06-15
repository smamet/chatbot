from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from chatbot.domain.models.fulfillment import FulfillmentKind
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
    test_sid = r.json().get("test_session")
    assert test_sid and test_sid.startswith("test:")
    r = client.get(f"/dashboard/bots/{slug}?tab=chat&test_session={test_sid}")
    assert "echo:hello dash" in r.text
    assert "chat-test.js" in r.text
    assert "markdown.js" in r.text

    r = client.post(
        f"/dashboard/bots/{slug}/chat-test/reset",
        data={"test_session": test_sid},
        follow_redirects=False,
    )
    assert r.status_code == 303
    r = client.get(f"/dashboard/bots/{slug}?tab=chat&test_session={test_sid}")
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
        data={"role": UserRole.CLIENT_OPERATOR.value},
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


def test_client_operator_cannot_export_import(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    with factory() as session:
        reader = UserService(SqlAlchemyUserRepository(session)).create_user(
            email="reader@test.com",
            password="read-pass",
            role=UserRole.CLIENT_OPERATOR,
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


def test_validation_tab_renders_quill_editor_for_email(dashboard_env) -> None:
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
        from chatbot.adapters.persistence.conversation_repository import (
            SqlAlchemyConversationRepository,
        )
        from chatbot.domain.models.message import ChatMessage, MessageRole

        SqlAlchemyConversationRepository(session, tenant_id).append_message(
            "email:client@example.com",
            ChatMessage(role=MessageRole.USER, content="Please send a quote"),
        )
        SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="**Hello** client",
            draft_html="<p><strong>Hello</strong> client</p>",
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.get(f"/dashboard/bots/{slug}?tab=validation")
    assert r.status_code == 200
    assert "validation-inbox-row" in r.text
    assert "validation-row-chevron" in r.text
    assert f"/dashboard/bots/{slug}/validation/" in r.text
    detail = client.get(f"/dashboard/bots/{slug}/validation/1")
    assert detail.status_code == 200
    assert "validation-quill" in detail.text
    assert "quill@2.0.3" in detail.text
    assert "/validation/1/save" in detail.text
    assert "Generated" in detail.text
    assert "Received" in detail.text
    assert "Re-resolve products" not in detail.text


def test_validation_retry_quote_clears_error(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )
        from chatbot.domain.models.integration import IntegrationType

        SqlAlchemyIntegrationRepository(session).create(
            tenant_id=tenant_id,
            type=IntegrationType.ERPNEXT,
            config={
                "url": "https://erp.example.com",
                "api_key": "k",
                "api_secret": "s",
                "allow_create_quotation": True,
                "allow_create_customer": True,
            },
            active=True,
        )
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Quote draft",
            fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
            quote_proposal_json=json.dumps(
                {"type": "quote.create", "lines": [{"product": "Widget", "qty": 1}]}
            ),
            quote_resolved_json=json.dumps(
                [
                    {
                        "requested_label": "Widget",
                        "qty": 1,
                        "item_code": "SKU-1",
                        "status": "resolved",
                        "rate": 10.0,
                    }
                ]
            ),
        )
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            pending.id,
            fulfillment_error="ERPNext customer creation failed (customer_create_failed): bad name",
        )
        session.commit()
        reply_id = pending.id

    _login(client, admin.email, "admin-pass")
    detail = client.get(f"/dashboard/bots/{slug}/validation/{reply_id}")
    assert detail.status_code == 200
    assert "validation-quote-retry-form" in detail.text
    assert "customer_create_failed" in detail.text

    with patch(
        "chatbot.interfaces.api.routers.dashboard_web.QuoteFulfillmentService"
    ) as svc_cls:
        svc_cls.return_value.retry_quote_fulfillment.side_effect = None
        r = client.post(
            f"/dashboard/bots/{slug}/validation/{reply_id}/retry-quote",
            follow_redirects=False,
        )
    assert r.status_code == 303
    svc_cls.return_value.retry_quote_fulfillment.assert_called_once()

    with factory() as session:
        from chatbot.application.validation_audit_service import ValidationAuditService
        from chatbot.domain.models.pending_reply_audit import ValidationAuditAction

        saved = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        activity = ValidationAuditService(session).list_activity(tenant_id, limit=10)
        assert any(e.action == ValidationAuditAction.RETRY_QUOTE for e in activity)
        assert saved is not None


def test_validation_retry_quote_requires_error(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Quote draft",
            fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
            quote_external_id="QTN-OK",
        )
        session.commit()
        reply_id = pending.id

    _login(client, admin.email, "admin-pass")
    r = client.post(f"/dashboard/bots/{slug}/validation/{reply_id}/retry-quote")
    assert r.status_code == 400


def test_validation_inbox_shows_quote_and_attachment_flags(dashboard_env) -> None:
    from chatbot.application.quote_pdf_storage import attachment_entry, encode_attachments_json
    from chatbot.adapters.persistence.pending_reply_repository import (
        SqlAlchemyPendingReplyRepository,
    )

    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        repo = SqlAlchemyPendingReplyRepository(session)
        plain = repo.create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:plain@example.com",
            channel="email",
            recipient_id="plain@example.com",
            draft_text="Plain reply",
        )
        quote = repo.create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:quote@example.com",
            channel="email",
            recipient_id="quote@example.com",
            draft_text="Quote draft",
            fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
            quote_external_id="QTN-0042",
        )
        attached = repo.create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:files@example.com",
            channel="email",
            recipient_id="files@example.com",
            draft_text="With files",
        )
        session.flush()
        att_dir = data / "attachments" / slug / str(attached.id)
        att_dir.mkdir(parents=True, exist_ok=True)
        att_path = att_dir / "spec.pdf"
        att_path.write_bytes(b"%PDF")
        repo.update_quote_fields(
            attached.id,
            attachments_json=encode_attachments_json(
                [attachment_entry(path=att_path, filename="spec.pdf")]
            ),
        )
        session.commit()
        plain_id, quote_id, attached_id = plain.id, quote.id, attached.id

    _login(client, admin.email, "admin-pass")
    inbox = client.get(f"/dashboard/bots/{slug}?tab=validation")
    assert inbox.status_code == 200
    assert "validation-inbox-flag--quote" not in _inbox_row_html(inbox.text, plain_id)
    assert "validation-inbox-flag--attach" not in _inbox_row_html(inbox.text, plain_id)
    assert "validation-inbox-flag--quote" in _inbox_row_html(inbox.text, quote_id)
    assert "QTN-0042" in _inbox_row_html(inbox.text, quote_id)
    assert "validation-inbox-flag--attach" in _inbox_row_html(inbox.text, attached_id)
    assert "1 file" in _inbox_row_html(inbox.text, attached_id)


def _inbox_row_html(page_html: str, reply_id: int) -> str:
    marker = f'data-href="/dashboard/bots/'
    start = page_html.find(f'{marker}')
    while start != -1:
        end = page_html.find("</tr>", start)
        row = page_html[start:end]
        if f"/validation/{reply_id}\"" in row:
            return row
        start = page_html.find(marker, end)
    raise AssertionError(f"inbox row for reply {reply_id} not found")


def test_validation_tab_renders_markdown_and_session_label(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.VALIDATION,
            config={"phone_number_id": "123", "access_token": "tok"},
        )
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="whatsapp:+33600000000",
            channel="whatsapp",
            recipient_id="+33600000000",
            draft_text="**Hello** client",
        )
        session.commit()
    _login(client, admin.email, "admin-pass")
    r = client.get(f"/dashboard/bots/{slug}?tab=validation")
    assert r.status_code == 200
    assert "+33600000000" in r.text
    detail = client.get(f"/dashboard/bots/{slug}/validation/1")
    assert detail.status_code == 200
    assert 'class="validation-message-body msg-body js-md"' in detail.text
    assert "**Hello** client" in detail.text
    assert "markdown.js" in detail.text


def test_validation_save_draft_updates_html_and_message(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        from chatbot.adapters.persistence.conversation_repository import (
            SqlAlchemyConversationRepository,
        )
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )
        from chatbot.domain.models.message import ChatMessage, MessageRole

        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Original body",
            draft_html="<p>Original body</p>",
        )
        SqlAlchemyConversationRepository(session, tenant_id).append_message(
            "email:client@example.com",
            ChatMessage(role=MessageRole.ASSISTANT, content="Original body"),
        )
        session.commit()
        reply_id = pending.id

    _login(client, admin.email, "admin-pass")
    r = client.post(
        f"/dashboard/bots/{slug}/validation/{reply_id}/save",
        data={"draft_html": "<p>Edited <strong>body</strong></p>"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].endswith(f"/validation/{reply_id}")

    with factory() as session:
        from chatbot.adapters.persistence.orm import PendingReplyEditRow
        from sqlalchemy import select

        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(reply_id)
        assert reply is not None
        assert reply.draft_html is not None
        assert "<strong>body</strong>" in reply.draft_html
        assert "Edited" in reply.draft_text
        edits = session.scalars(select(PendingReplyEditRow)).all()
        assert len(edits) == 1


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


def _save_erpnext_integration(client: TestClient, slug: str, **extra: str) -> None:
    data = {
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
    }
    data.update(extra)
    r = client.post(f"/dashboard/bots/{slug}/integrations", data=data, follow_redirects=False)
    assert r.status_code == 303


def test_chat_test_works_without_identity_when_integration_active(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug)
    test_sid = "test:00000000-0000-4000-8000-000000000001"
    r = client.post(
        f"/dashboard/bots/{slug}/chat-test/send",
        data={"message": "hello", "test_session": test_sid},
    )
    assert r.status_code == 200
    with factory() as session:
        from chatbot.adapters.persistence.conversation_repository import (
            SqlAlchemyConversationRepository,
        )

        msgs = SqlAlchemyConversationRepository(session, tenant_id).list_messages(
            test_sid, limit=10
        )
        assert len(msgs) == 2


def test_test_chat_does_not_load_legacy_dashboard_session(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    with factory() as session:
        from chatbot.adapters.persistence.conversation_repository import (
            SqlAlchemyConversationRepository,
        )
        from chatbot.domain.models.message import ChatMessage, MessageRole

        repo = SqlAlchemyConversationRepository(session, tenant_id)
        legacy_sid = f"dashboard:{admin.id}"
        repo.append_message(legacy_sid, ChatMessage(role=MessageRole.USER, content="old"))
        repo.append_message(legacy_sid, ChatMessage(role=MessageRole.ASSISTANT, content="legacy"))
        session.commit()
    r = client.get(f"/dashboard/bots/{slug}?tab=chat")
    assert r.status_code == 200
    assert "old" not in r.text or '"role": "user", "content": "old"' not in r.text
    assert "chat-initial" in r.text
    assert '"[]"' in r.text or "[]" in r.text.split("chat-initial")[1][:200]


def test_chat_test_lists_previous_sessions(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    client.post(
        f"/dashboard/bots/{slug}/chat-test/send",
        data={"message": "hello", "test_email": "client@example.com"},
    )
    r = client.get(f"/dashboard/bots/{slug}?tab=chat")
    assert r.status_code == 200
    assert "Sessions" in r.text
    assert "history-layout" in r.text
    assert "client@example.com" in r.text
    with factory() as session:
        from chatbot.adapters.persistence.test_chat_session_repository import (
            TestChatSessionRepository,
        )

        rows = TestChatSessionRepository(session, tenant_id).list_recent()
        assert len(rows) == 1
        assert rows[0].session_id == "email:client@example.com"


def test_chat_test_uses_identity_session_id(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug)
    r = client.post(
        f"/dashboard/bots/{slug}/chat-test/send",
        data={"message": "hello", "test_email": "client@example.com"},
    )
    assert r.status_code == 200
    with factory() as session:
        from chatbot.adapters.persistence.conversation_repository import (
            SqlAlchemyConversationRepository,
        )

        msgs = SqlAlchemyConversationRepository(session, tenant_id).list_messages(
            "email:client@example.com", limit=10
        )
        assert len(msgs) == 2


def test_create_customer_endpoint_respects_permission(dashboard_env) -> None:
    client, admin, _, slug, *_ = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug)
    r = client.post(
        f"/dashboard/bots/{slug}/integrations/erpnext/create-customer",
        data={
            "integration_type": "erpnext",
            "url": "https://erp.example.com",
            "api_key": "test-key",
            "api_secret": "test-secret",
            "test_email": "new@example.com",
        },
    )
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "creation_disabled"

    with patch("chatbot.interfaces.api.routers.dashboard_web.create_erpnext_customer_for_test") as mock_create:
        mock_create.return_value = {
            "ok": True,
            "message": "Customer created: New Corp",
            "customer": "New Corp",
            "created": True,
        }
        r = client.post(
            f"/dashboard/bots/{slug}/integrations/erpnext/create-customer",
            data={
                "integration_type": "erpnext",
                "url": "https://erp.example.com",
                "api_key": "test-key",
                "api_secret": "test-secret",
                "allow_create_customer": "on",
                "test_email": "new@example.com",
            },
        )
    body = r.json()
    assert body["ok"] is True
    assert body["customer"] == "New Corp"


def test_create_quotation_endpoint_respects_permission(dashboard_env) -> None:
    client, admin, _, slug, *_ = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug)
    r = client.post(
        f"/dashboard/bots/{slug}/integrations/erpnext/create-quotation",
        data={
            "integration_type": "erpnext",
            "url": "https://erp.example.com",
            "api_key": "test-key",
            "api_secret": "test-secret",
            "test_email": "client@example.com",
            "item_code": "SKU-1",
            "qty": "1",
        },
    )
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "creation_disabled"

    with patch("chatbot.interfaces.api.routers.dashboard_web.create_erpnext_quotation_for_test") as mock_create:
        mock_create.return_value = {
            "ok": True,
            "message": "Quotation created: QTN-0001",
            "customer": "Client Corp",
            "quote_name": "QTN-0001",
            "pdf_url": f"/dashboard/bots/{slug}/integrations/erpnext/quotation-pdf/QTN-0001",
        }
        r = client.post(
            f"/dashboard/bots/{slug}/integrations/erpnext/create-quotation",
            data={
                "integration_type": "erpnext",
                "url": "https://erp.example.com",
                "api_key": "test-key",
                "api_secret": "test-secret",
                "allow_create_quotation": "on",
                "test_email": "client@example.com",
                "item_code": "SKU-1",
                "qty": "2",
            },
        )
    body = r.json()
    assert body["ok"] is True
    assert body["quote_name"] == "QTN-0001"


def test_create_quotation_stream_ndjson(dashboard_env) -> None:
    client, admin, _, slug, *_ = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug, allow_create_quotation="on")

    def fake_create(*_args, on_log=None, **_kwargs):
        if on_log is not None:
            on_log.step("Looking up customer…")
            on_log.step("Quotation created: QTN-STREAM")
        return {
            "ok": True,
            "message": "Quotation created: QTN-STREAM",
            "customer": "Client Corp",
            "quote_name": "QTN-STREAM",
            "pdf_url": None,
            "pdf_filename": None,
            "pdf_warning": None,
        }

    with patch(
        "chatbot.interfaces.api.routers.dashboard_web.create_erpnext_quotation_for_test",
        side_effect=fake_create,
    ):
        r = client.post(
            f"/dashboard/bots/{slug}/integrations/erpnext/create-quotation",
            data={
                "integration_type": "erpnext",
                "url": "https://erp.example.com",
                "api_key": "test-key",
                "api_secret": "test-secret",
                "allow_create_quotation": "on",
                "stream": "1",
                "test_email": "client@example.com",
                "item_code": "SKU-1",
                "qty": "1",
            },
        )

    assert r.status_code == 200
    assert "application/x-ndjson" in r.headers.get("content-type", "")
    events = [json.loads(line) for line in r.text.strip().split("\n") if line.strip()]
    assert events[0] == {"event": "log", "message": "Looking up customer…"}
    assert events[1] == {"event": "log", "message": "Quotation created: QTN-STREAM"}
    assert events[-1]["event"] == "done"
    assert events[-1]["ok"] is True
    assert events[-1]["quote_name"] == "QTN-STREAM"


def test_chat_test_auto_creates_quote_when_resolved(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug, allow_create_quotation="on")
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "out",
            "mode": "validation",
            "active": "on",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
            "username": "bot@example.com",
            "password": "secret",
            "from_addr": "bot@example.com",
        },
        follow_redirects=False,
    )

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    original_run = dash_mod._run_dashboard_chat

    def _hook_run(request, settings, tenant, user, message, session, *, test_email="", test_phone="", test_session=""):
        session_id = dash_mod._dashboard_chat_session_id(
            user,
            test_email=test_email,
            test_phone=test_phone,
            require_identity=False,
        )
        result = SimpleNamespace(
            text="Your quote is ready.",
            usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            hook_type="quote.create",
            hook_payload_json='{"type":"quote.create","lines":[{"product":"Widget","qty":1}]}',
            hook_event_id=None,
        )
        return session_id, result

    created = SimpleNamespace(
        quote_name="QTN-TEST",
        pdf_url=f"/dashboard/bots/{slug}/integrations/erpnext/quotation-pdf/QTN-TEST",
        pdf_filename="QTN-TEST.pdf",
        pdf_warning=None,
    )

    dash_mod._run_dashboard_chat = _hook_run  # type: ignore[assignment]
    try:
        with patch(
            "chatbot.interfaces.api.routers.dashboard_web.resolve_quote_hook",
            return_value=(
                SimpleNamespace(lines=(SimpleNamespace(product="Widget", qty=1, item_code=None),), notes=None),
                '[{"requested_label":"Widget","qty":1,"item_code":"SKU-1","status":"resolved"}]',
            ),
        ), patch(
            "chatbot.interfaces.api.routers.dashboard_web.create_quote_for_session",
            return_value=created,
        ):
            r = client.post(
                f"/dashboard/bots/{slug}/chat-test/send",
                data={
                    "message": "I need a quote",
                    "test_email": "client@example.com",
                },
            )
    finally:
        dash_mod._run_dashboard_chat = original_run  # type: ignore[assignment]

    assert r.status_code == 200
    body = r.json()
    assert body["queued"] is False
    assert body["pdf_url"] == created.pdf_url
    assert body["quote_name"] == "QTN-TEST"
    with factory() as session:
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id)
        assert pending == []


def test_chat_test_queues_quote_hook_when_unresolved(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug, allow_create_quotation="on")
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "out",
            "mode": "validation",
            "active": "on",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
            "username": "bot@example.com",
            "password": "secret",
            "from_addr": "bot@example.com",
        },
        follow_redirects=False,
    )

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    original_run = dash_mod._run_dashboard_chat

    def _hook_run(request, settings, tenant, user, message, session, *, test_email="", test_phone="", test_session=""):
        session_id = dash_mod._dashboard_chat_session_id(
            user,
            test_email=test_email,
            test_phone=test_phone,
            require_identity=False,
        )
        result = SimpleNamespace(
            text="Your quote is ready for review.",
            usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            hook_type="quote.create",
            hook_payload_json='{"type":"quote.create","lines":[{"product":"Widget","qty":1}]}',
            hook_event_id=None,
        )
        return session_id, result

    dash_mod._run_dashboard_chat = _hook_run  # type: ignore[assignment]
    try:
        with patch(
            "chatbot.application.outbound_orchestrator.resolved_lines_to_json",
            return_value='[{"requested_label":"Widget","qty":1,"status":"ambiguous"}]',
        ):
            r = client.post(
                f"/dashboard/bots/{slug}/chat-test/send",
                data={
                    "message": "I need a quote",
                    "test_email": "client@example.com",
                },
            )
    finally:
        dash_mod._run_dashboard_chat = original_run  # type: ignore[assignment]

    assert r.status_code == 200
    body = r.json()
    assert body["queued"] is True
    assert body["hook_type"] == "quote.create"
    with factory() as session:
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id)
        assert len(pending) == 1
        assert pending[0].session_id == "email:client@example.com"


def test_chat_test_simulates_email_channel_queues_plain_reply(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "out",
            "mode": "validation",
            "active": "on",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
            "username": "bot@example.com",
            "password": "secret",
            "from_addr": "bot@example.com",
        },
        follow_redirects=False,
    )

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    original_run = dash_mod._run_dashboard_chat

    def _plain_run(request, settings, tenant, user, message, session, *, test_email="", test_phone="", test_session=""):
        session_id = dash_mod._dashboard_chat_session_id(
            user,
            test_email=test_email,
            test_phone=test_phone,
            require_identity=False,
        )
        result = SimpleNamespace(
            text="Here is the information you requested.",
            usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            hook_type=None,
            hook_payload_json=None,
            hook_event_id=None,
        )
        return session_id, result

    dash_mod._run_dashboard_chat = _plain_run  # type: ignore[assignment]
    try:
        r = client.post(
            f"/dashboard/bots/{slug}/chat-test/send",
            data={
                "message": "Tell me about deployment",
                "test_email": "client@example.com",
                "channel": "email",
            },
        )
    finally:
        dash_mod._run_dashboard_chat = original_run  # type: ignore[assignment]

    assert r.status_code == 200
    body = r.json()
    assert body["queued"] is True
    assert body["hook_type"] is None
    assert "email" in (body["message"] or "").lower()
    with factory() as session:
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id)
        assert len(pending) == 1
        assert pending[0].channel == "email"
        assert pending[0].draft_html is not None


def test_chat_test_simulates_email_channel_queues_with_draft_html(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug, allow_create_quotation="on")
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "email",
            "direction": "out",
            "mode": "validation",
            "active": "on",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
            "username": "bot@example.com",
            "password": "secret",
            "from_addr": "bot@example.com",
        },
        follow_redirects=False,
    )

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    original_run = dash_mod._run_dashboard_chat

    def _hook_run(request, settings, tenant, user, message, session, *, test_email="", test_phone="", test_session=""):
        session_id = dash_mod._dashboard_chat_session_id(
            user,
            test_email=test_email,
            test_phone=test_phone,
            require_identity=False,
        )
        result = SimpleNamespace(
            text="Your quote is ready.",
            usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            hook_type="quote.create",
            hook_payload_json='{"type":"quote.create","lines":[{"product":"Widget","qty":1}]}',
            hook_event_id=None,
        )
        return session_id, result

    dash_mod._run_dashboard_chat = _hook_run  # type: ignore[assignment]
    try:
        with patch(
            "chatbot.interfaces.api.routers.dashboard_web.resolve_quote_hook",
            return_value=(
                SimpleNamespace(lines=(SimpleNamespace(product="Widget", qty=1, item_code=None),), notes=None),
                '[{"requested_label":"Widget","qty":1,"item_code":"SKU-1","status":"resolved"}]',
            ),
        ), patch(
            "chatbot.interfaces.api.routers.dashboard_web.create_quote_for_session",
        ) as create_mock:
            r = client.post(
                f"/dashboard/bots/{slug}/chat-test/send",
                data={
                    "message": "I need a quote",
                    "test_email": "client@example.com",
                    "channel": "email",
                },
            )
    finally:
        dash_mod._run_dashboard_chat = original_run  # type: ignore[assignment]

    assert r.status_code == 200
    body = r.json()
    create_mock.assert_not_called()
    assert body["queued"] is True
    assert body["pdf_url"] is None
    assert "email" in (body["message"] or "").lower()
    with factory() as session:
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id)
        assert len(pending) == 1
        assert pending[0].channel == "email"
        assert pending[0].draft_html is not None
        assert "<" in pending[0].draft_html


def test_chat_test_simulated_email_requires_email_connector(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    client.post(
        f"/dashboard/bots/{slug}/connectors",
        data={
            "connector_type": "whatsapp",
            "direction": "out",
            "mode": "validation",
            "active": "on",
            "phone_number_id": "123",
            "access_token": "tok",
        },
        follow_redirects=False,
    )

    import chatbot.interfaces.api.routers.dashboard_web as dash_mod

    original_run = dash_mod._run_dashboard_chat

    def _hook_run(request, settings, tenant, user, message, session, *, test_email="", test_phone="", test_session=""):
        return "email:client@example.com", SimpleNamespace(
            text="Reply",
            hook_type="quote.create",
            hook_payload_json='{"type":"quote.create","lines":[{"product":"X","qty":1}]}',
            hook_event_id=None,
        )

    dash_mod._run_dashboard_chat = _hook_run  # type: ignore[assignment]
    try:
        r = client.post(
            f"/dashboard/bots/{slug}/chat-test/send",
            data={
                "message": "quote",
                "test_email": "client@example.com",
                "channel": "email",
            },
        )
    finally:
        dash_mod._run_dashboard_chat = original_run  # type: ignore[assignment]

    assert r.status_code == 200
    body = r.json()
    assert body["queued"] is False
    assert "email" in (body["message"] or "").lower()
    with factory() as session:
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        assert SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id) == []


def test_sync_catalog_endpoint_starts_background_job(dashboard_env) -> None:
    client, admin, _, slug, *_ = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug)
    with patch(
        "chatbot.interfaces.api.routers.dashboard_web._run_catalog_sync_background"
    ) as mock_run:
        r = client.post(f"/dashboard/bots/{slug}/integrations/erpnext/sync-catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "background" in body["message"].lower()
    mock_run.assert_called_once()


def test_purge_catalog_endpoint(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    _login(client, admin.email, "admin-pass")
    _save_erpnext_integration(client, slug)
    catalog_dir = data / "catalog" / slug
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "item.md").write_text("# item", encoding="utf-8")
    r = client.post(f"/dashboard/bots/{slug}/integrations/erpnext/purge-catalog")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert not list(catalog_dir.glob("*.md"))


def test_upload_documents_auto_syncs(dashboard_env) -> None:
    from chatbot.application.sync_service import IngestSyncService

    client, admin, _, slug, _, data, _ = dashboard_env
    _login(client, admin.email, "admin-pass")

    with patch.object(
        IngestSyncService,
        "reconcile_root",
        return_value=["ingested 1 chunks: new.md"],
    ) as mock_reconcile:
        r = client.post(
            f"/dashboard/bots/{slug}/documents",
            files=[("files", ("new.md", b"# New doc\n", "text/markdown"))],
            follow_redirects=False,
        )

    assert r.status_code == 303
    mock_reconcile.assert_called_once()
    assert mock_reconcile.call_args.kwargs == {"fresh": False}
    assert (data / "docs" / slug / "new.md").is_file()
    r = client.get(f"/dashboard/bots/{slug}?tab=documents")
    assert "ingested 1 chunks: new.md" in r.text


def test_delete_document_auto_syncs(dashboard_env) -> None:
    from chatbot.application.sync_service import IngestSyncService

    client, admin, _, slug, _, data, _ = dashboard_env
    _login(client, admin.email, "admin-pass")
    docs_dir = data / "docs" / slug
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "gone.md").write_text("# gone\n", encoding="utf-8")

    with patch.object(
        IngestSyncService,
        "reconcile_root",
        return_value=["pruned missing: gone.md"],
    ) as mock_reconcile:
        r = client.post(
            f"/dashboard/bots/{slug}/documents/delete",
            data={"path": "gone.md"},
            follow_redirects=False,
        )

    assert r.status_code == 303
    mock_reconcile.assert_called_once()
    assert mock_reconcile.call_args.kwargs == {"fresh": False}
    assert not (docs_dir / "gone.md").is_file()
    r = client.get(f"/dashboard/bots/{slug}?tab=documents")
    assert "pruned missing: gone.md" in r.text


def test_validation_detail_upload_and_delete_attachment(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Hello",
        )
        session.commit()
        reply_id = pending.id

    _login(client, admin.email, "admin-pass")
    detail = client.get(f"/dashboard/bots/{slug}/validation/{reply_id}")
    assert detail.status_code == 200
    assert "validation-attachments-dropzone" in detail.text

    upload = client.post(
        f"/dashboard/bots/{slug}/validation/{reply_id}/attachments",
        files=[("files", ("extra.pdf", b"%PDF-test", "application/pdf"))],
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["ok"] is True
    assert len(body["attachments"]) == 1
    stored_path = body["attachments"][0]["path"]
    assert Path(stored_path).is_file()

    delete = client.delete(
        f"/dashboard/bots/{slug}/validation/{reply_id}/attachments",
        params={"path": stored_path},
    )
    assert delete.status_code == 200
    assert delete.json()["attachments"] == []
    assert not Path(stored_path).is_file()


def test_approve_non_quote_email_with_attachment(dashboard_env) -> None:
    from chatbot.application.quote_pdf_storage import (
        attachment_entry,
        encode_attachments_json,
    )

    client, admin, _, slug, tenant_id, data, factory = dashboard_env
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

        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="See attachment",
        )
        session.flush()
        reply_id = pending.id
        att_dir = data / "attachments" / slug / str(reply_id)
        att_dir.mkdir(parents=True, exist_ok=True)
        att_path = att_dir / "file.pdf"
        att_path.write_bytes(b"%PDF")
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            reply_id,
            attachments_json=encode_attachments_json(
                [attachment_entry(path=att_path, filename="file.pdf")]
            ),
        )
        session.commit()

    _login(client, admin.email, "admin-pass")
    with patch("chatbot.application.channel_outbound.send_email_reply") as send_mock:
        r = client.post(
            f"/dashboard/bots/{slug}/validation/{reply_id}/approve",
            follow_redirects=False,
        )
    assert r.status_code == 303
    send_mock.assert_called_once()
    assert send_mock.call_args.kwargs["attachments"][0].filename == "file.pdf"
    assert not att_path.is_file()


def test_validation_detail_shows_stale_banner_when_quote_changed(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    with factory() as session:
        from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )
        from chatbot.domain.models.integration import IntegrationType

        SqlAlchemyIntegrationRepository(session).create(
            tenant_id=tenant_id,
            type=IntegrationType.ERPNEXT,
            config={
                "url": "https://erp.example.com",
                "api_key": "k",
                "api_secret": "s",
                "allow_create_quotation": True,
            },
            active=True,
        )
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Quote draft",
            fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
            quote_external_id="QTN-STALE",
            quote_resolved_json=json.dumps(
                [
                    {
                        "requested_label": "Widget",
                        "qty": 1,
                        "item_code": "SKU-1",
                        "status": "resolved",
                        "rate": 10.0,
                    }
                ]
            ),
        )
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            pending.id,
            quote_erp_modified="2026-06-15 14:17:39",
        )
        session.commit()
        reply_id = pending.id

    _login(client, admin.email, "admin-pass")
    with patch(
        "chatbot.interfaces.api.routers.dashboard_web.erpnext_integration_for_tenant"
    ) as integration_mock:
        erp_client = MagicMock()
        erp_client.get_quotation.return_value = {"modified": "2026-06-15 14:18:15"}
        integration_mock.return_value = (erp_client, {"url": "https://erp.example.com"})
        detail = client.get(f"/dashboard/bots/{slug}/validation/{reply_id}")

    assert detail.status_code == 200
    assert "validation-quote-stale-banner" in detail.text
    assert "Proceed &amp; send" in detail.text
    assert "Approve &amp; send" not in detail.text


def test_approve_quote_redirects_with_warning_when_stale_unconfirmed(dashboard_env) -> None:
    client, admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )
        from chatbot.domain.models.integration import IntegrationType

        SqlAlchemyIntegrationRepository(session).create(
            tenant_id=tenant_id,
            type=IntegrationType.ERPNEXT,
            config={"url": "https://erp.example.com", "api_key": "k", "api_secret": "s"},
            active=True,
        )
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Quote draft",
            fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
            quote_external_id="QTN-STALE",
            quote_resolved_json=json.dumps(
                [
                    {
                        "requested_label": "Widget",
                        "qty": 1,
                        "item_code": "SKU-1",
                        "status": "resolved",
                    }
                ]
            ),
        )
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            pending.id,
            quote_erp_modified="2026-06-15 14:17:39",
        )
        session.commit()
        reply_id = pending.id

    _login(client, admin.email, "admin-pass")
    with patch(
        "chatbot.interfaces.api.routers.dashboard_web.erpnext_integration_for_tenant"
    ) as integration_mock, patch(
        "chatbot.interfaces.api.routers.dashboard_web.QuoteFulfillmentService"
    ) as fulfill_cls:
        erp_client = MagicMock()
        erp_client.get_quotation.return_value = {"modified": "2026-06-15 14:18:15"}
        integration_mock.return_value = (erp_client, {"url": "https://erp.example.com"})
        r = client.post(
            f"/dashboard/bots/{slug}/validation/{reply_id}/approve",
            follow_redirects=False,
        )
    assert r.status_code == 303
    fulfill_cls.assert_not_called()
    follow = client.get(f"/dashboard/bots/{slug}/validation/{reply_id}")
    assert "validation-warning" in follow.text


def test_validation_attachment_file_view(dashboard_env) -> None:
    from chatbot.application.quote_pdf_storage import attachment_entry, encode_attachments_json

    client, admin, _, slug, tenant_id, data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Draft",
        )
        session.flush()
        reply_id = pending.id
        att_dir = data / "attachments" / slug / str(reply_id)
        att_dir.mkdir(parents=True, exist_ok=True)
        att_path = att_dir / "notes.txt"
        att_path.write_text("hello", encoding="utf-8")
        SqlAlchemyPendingReplyRepository(session).update_quote_fields(
            reply_id,
            attachments_json=encode_attachments_json(
                [attachment_entry(path=att_path, filename="notes.txt", mime_type="text/plain")]
            ),
        )
        session.commit()

    _login(client, admin.email, "admin-pass")
    ok = client.get(
        f"/dashboard/bots/{slug}/validation/{reply_id}/attachments/file",
        params={"path": str(att_path)},
    )
    assert ok.status_code == 200
    assert ok.content == b"hello"

    bad = client.get(
        f"/dashboard/bots/{slug}/validation/{reply_id}/attachments/file",
        params={"path": "/etc/passwd"},
    )
    assert bad.status_code == 404

    detail = client.get(f"/dashboard/bots/{slug}/validation/{reply_id}")
    assert 'href="/dashboard/bots/' in detail.text
    assert "notes.txt" in detail.text


def _create_operator(factory, tenant_id: int):
    with factory() as session:
        op = UserService(SqlAlchemyUserRepository(session)).create_user(
            email="operator@test.com",
            password="op-pass",
            role=UserRole.CLIENT_OPERATOR,
        )
        UserService(SqlAlchemyUserRepository(session)).grant_access(op.id, tenant_id)
        session.commit()
        return op


def test_client_operator_login_single_bot_goes_to_validation(dashboard_env) -> None:
    client, _admin, _, slug, tenant_id, _data, factory = dashboard_env
    _create_operator(factory, tenant_id)
    r = client.post(
        "/auth/login",
        data={"email": "operator@test.com", "password": "op-pass"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/dashboard/bots/{slug}?tab=validation"


def test_client_operator_redirects_config_tab_to_validation(dashboard_env) -> None:
    client, _admin, _, slug, tenant_id, _data, factory = dashboard_env
    _create_operator(factory, tenant_id)
    _login(client, "operator@test.com", "op-pass")
    r = client.get(f"/dashboard/bots/{slug}?tab=config", follow_redirects=False)
    assert r.status_code == 303
    assert "tab=validation" in r.headers["location"]


def test_client_operator_can_validate_email_reply(dashboard_env) -> None:
    client, _admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
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
            draft_text="**Hello**",
            draft_html="<p><strong>Hello</strong></p>",
        )
        session.commit()
    _create_operator(factory, tenant_id)
    _login(client, "operator@test.com", "op-pass")
    detail = client.get(f"/dashboard/bots/{slug}/validation/1")
    assert detail.status_code == 200
    assert "validation-quill" in detail.text
    assert "Approve &amp; send" in detail.text or "Approve & send" in detail.text


def test_client_operator_reject_creates_audit(dashboard_env) -> None:
    client, _admin, _, slug, tenant_id, _data, factory = dashboard_env
    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.VALIDATION,
            config={},
        )
        from chatbot.adapters.persistence.pending_reply_repository import (
            SqlAlchemyPendingReplyRepository,
        )

        SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id="whatsapp:+1",
            channel="whatsapp",
            recipient_id="+1",
            draft_text="Hi",
        )
        session.commit()
    _create_operator(factory, tenant_id)
    _login(client, "operator@test.com", "op-pass")
    r = client.post(f"/dashboard/bots/{slug}/validation/1/reject", follow_redirects=False)
    assert r.status_code == 303
    with factory() as session:
        from chatbot.application.validation_audit_service import ValidationAuditService

        saved = SqlAlchemyPendingReplyRepository(session).find_by_id(1)
        assert saved is not None
        assert saved.status.value == "rejected"
        assert saved.resolved_by == "operator@test.com"
        activity = ValidationAuditService(session).list_activity(tenant_id, limit=10)
        assert any(e.action == "rejected" for e in activity)
    inbox = client.get(f"/dashboard/bots/{slug}?tab=validation&vsub=rejected")
    assert inbox.status_code == 200
    assert "validation-inbox-row" in inbox.text
