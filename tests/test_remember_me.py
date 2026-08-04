from __future__ import annotations

import re

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.orm import UserRow
from evenor.adapters.persistence.user_repository import SqlAlchemyUserRepository
from evenor.application.remember_me_service import REMEMBER_COOKIE_NAME, RememberMeService
from evenor.application.user_service import UserService
from evenor.config.settings import reset_settings_cache_for_tests
from evenor.domain.models.user import UserRole
from evenor.interfaces.api.main import create_app, refresh_genai_clients_if_needed


@pytest.fixture
def remember_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    data = tmp_path / "data"
    db = data / "remember.db"
    secret = Fernet.generate_key().decode()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LANCEDB_ROOT", str(data / "lancedb"))
    monkeypatch.setenv("APP_SECRET_KEY", secret)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("DEV_MODE", "true")
    reset_settings_cache_for_tests()
    from evenor.config.settings import get_settings

    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        user_svc = UserService(SqlAlchemyUserRepository(session))
        admin = user_svc.create_user(
            email="admin@test.com", password="admin-pass", role=UserRole.ADMIN
        )
        operator = user_svc.create_user(
            email="op@test.com", password="op-pass", role=UserRole.CLIENT_OPERATOR
        )
        session.commit()
    app = create_app()
    app.state.session_factory = factory
    refresh_genai_clients_if_needed(app)
    with TestClient(app) as client:
        yield client, admin, operator, factory, settings
    reset_settings_cache_for_tests()
    engine.dispose()


def _set_cookie_header(response) -> str:
    return response.headers.get("set-cookie", "")


def _cookie_names(header: str) -> list[str]:
    return re.findall(r"^([^=]+)=", header, flags=re.MULTILINE)


def test_login_with_remember_me_sets_remember_cookie(remember_env) -> None:
    client, admin, _, _, _ = remember_env
    r = client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass", "remember_me": "on"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    cookies = _cookie_names(_set_cookie_header(r))
    assert REMEMBER_COOKIE_NAME in cookies
    assert client.cookies.get(REMEMBER_COOKIE_NAME)
    assert client.cookies.get("session")


def test_login_without_remember_clears_remember_cookie(remember_env) -> None:
    client, admin, _, _, _ = remember_env
    client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass", "remember_me": "on"},
        follow_redirects=False,
    )
    r = client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass"},
        follow_redirects=False,
    )
    header = _set_cookie_header(r).lower()
    assert "evenor_remember=" in header
    assert "max-age=0" in header or 'evenor_remember="";' in header or "expires=" in header


def test_remember_cookie_restores_session(remember_env) -> None:
    client, admin, _, factory, settings = remember_env
    login = client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass", "remember_me": "on"},
        follow_redirects=False,
    )
    remember_value = client.cookies.get(REMEMBER_COOKIE_NAME)
    assert remember_value
    client.cookies.clear()
    client.cookies.set(REMEMBER_COOKIE_NAME, remember_value)
    r = client.get("/dashboard/bots", follow_redirects=False)
    assert r.status_code == 200
    assert client.cookies.get("session")


def test_logout_revokes_remember_token(remember_env) -> None:
    client, admin, _, factory, _ = remember_env
    client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass", "remember_me": "on"},
        follow_redirects=False,
    )
    client.post("/auth/logout", follow_redirects=False)
    with factory() as session:
        row = session.get(UserRow, admin.id)
        assert row is not None
        assert row.remember_token_hash is None


def test_set_password_revokes_remember_token(remember_env) -> None:
    client, admin, _, factory, settings = remember_env
    with factory() as session:
        remember = RememberMeService(SqlAlchemyUserRepository(session), settings=settings)
        remember.issue_token(admin.id)
        session.commit()
    with factory() as session:
        user_svc = UserService(SqlAlchemyUserRepository(session))
        user_svc.set_password(admin.email, "new-pass")
        session.commit()
    with factory() as session:
        row = session.get(UserRow, admin.id)
        assert row is not None
        assert row.remember_token_hash is None


def test_logged_in_root_redirects_to_dashboard(remember_env) -> None:
    client, admin, _, _, _ = remember_env
    client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass"},
        follow_redirects=False,
    )
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/dashboard/bots"


def test_logged_in_login_form_redirects(remember_env) -> None:
    client, admin, _, _, _ = remember_env
    client.post(
        "/auth/login",
        data={"email": admin.email, "password": "admin-pass"},
        follow_redirects=False,
    )
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/dashboard/bots"


def test_anonymous_root_redirects_to_login(remember_env) -> None:
    client, _, _, _, _ = remember_env
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/auth/login"
