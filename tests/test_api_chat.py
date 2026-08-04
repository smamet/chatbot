from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.tenant_service import TenantService
from evenor.config.settings import reset_settings_cache_for_tests
from evenor.interfaces.api.deps import get_chat_service
from evenor.interfaces.api.main import create_app, refresh_genai_clients_if_needed


def test_healthz() -> None:
    client = TestClient(create_app())
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_redirects_to_login() -> None:
    client = TestClient(create_app())
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/auth/login"


def test_dashboard_redirects_to_bots() -> None:
    client = TestClient(create_app())
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/dashboard/bots"


@pytest.fixture
def tenant_chat_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    db = tmp_path / "api.db"
    data = tmp_path / "data"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LANCEDB_ROOT", str(data / "lancedb"))
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    from evenor.config.settings import get_settings

    reset_settings_cache_for_tests()
    engine = create_db_engine(get_settings(), for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        result = TenantService(SqlAlchemyTenantRepository(session)).create_tenant(
            name="API Test", slug="api-test"
        )
        session.commit()
        slug, token = result.tenant.slug, result.token
    app = create_app()
    app.state.session_factory = session_factory(engine)
    refresh_genai_clients_if_needed(app)

    class _FakeChatService:
        def handle_user_message(self, session_id: str, message: str, *, attachments=None):
            _ = attachments
            return SimpleNamespace(
                text="ok",
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    candidates_tokens=1,
                    total_tokens=2,
                ),
            )

    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService()
    with TestClient(app) as client:
        yield client, slug, token
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()
    engine.dispose()


def _chat_form(**fields: str) -> dict[str, str]:
    return {"session_id": fields.get("session_id", "s1"), "message": fields.get("message", "hi")}


def test_tenant_chat_requires_bearer(tenant_chat_client) -> None:
    client, slug, _token = tenant_chat_client
    r = client.post(f"/c/{slug}/chat", data=_chat_form())
    assert r.status_code == 401


def test_tenant_chat_rejects_wrong_bearer(tenant_chat_client) -> None:
    client, slug, _token = tenant_chat_client
    r = client.post(
        f"/c/{slug}/chat",
        data=_chat_form(),
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_tenant_chat_accepts_bearer(tenant_chat_client) -> None:
    client, slug, token = tenant_chat_client
    r = client.post(
        f"/c/{slug}/chat",
        data=_chat_form(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "ok"
    assert body["usage"]["total_tokens"] == 2


def test_tenant_chat_token_mismatch_slug_returns_403(tenant_chat_client) -> None:
    client, _slug, token = tenant_chat_client
    r = client.post(
        "/c/other-slug/chat",
        data=_chat_form(),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_tenant_chat_accepts_file_attachment(tenant_chat_client) -> None:
    client, slug, token = tenant_chat_client
    r = client.post(
        f"/c/{slug}/chat",
        data=_chat_form(message="see file"),
        files=[("files", ("doc.pdf", b"%PDF-1.4", "application/pdf"))],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
