from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

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

    r = client.post(
        f"/dashboard/bots/{slug}/chat-test/send",
        data={"message": "hello dash"},
    )
    assert r.status_code == 200
    assert r.json()["reply"] == "echo:hello dash"
    r = client.get(f"/dashboard/bots/{slug}?tab=chat")
    assert "echo:hello dash" in r.text
    assert "chat-test.js" in r.text

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
        repo.append_message("hist-1", ChatMessage(role=MessageRole.USER, content="x"))
        session.commit()

    r = client.get(f"/dashboard/bots/{slug}?tab=history&sid=hist-1")
    assert r.status_code == 200
    assert "hist-1" in r.text


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
