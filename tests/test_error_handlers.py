from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from fastapi.testclient import TestClient

from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.interfaces.api.main import create_app, refresh_genai_clients_if_needed


@pytest.fixture
def error_app(monkeypatch: pytest.MonkeyPatch, tmp_path):
    data = tmp_path / "data"
    secret = Fernet.generate_key().decode()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{data / 'err.db'}")
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LANCEDB_ROOT", str(data / "lancedb"))
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    monkeypatch.setenv("APP_SECRET_KEY", secret)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    reset_settings_cache_for_tests()
    app = create_app()

    @app.get("/dashboard/__test_error__")
    def _raise_unhandled() -> None:
        raise ValueError("boom test error")

    @app.get("/dashboard/__test_http_error__")
    def _raise_http() -> None:
        raise HTTPException(status_code=404, detail="Missing bot")

    @app.get("/api/__test_error__")
    def _raise_api() -> None:
        raise ValueError("api boom error")

    refresh_genai_clients_if_needed(app)
    yield app
    reset_settings_cache_for_tests()


def test_dashboard_error_page_shows_traceback_in_dev(error_app, monkeypatch) -> None:
    monkeypatch.setenv("DEV_MODE", "true")
    reset_settings_cache_for_tests()
    refresh_genai_clients_if_needed(error_app)

    client = TestClient(error_app, raise_server_exceptions=False)
    response = client.get("/dashboard/__test_error__")

    assert response.status_code == 500
    assert "Server error" in response.text
    assert "boom test error" in response.text
    assert "ValueError" in response.text
    assert "Traceback" in response.text
    assert "Back to bots" in response.text


def test_dashboard_error_page_hides_traceback_when_not_dev(error_app, monkeypatch) -> None:
    monkeypatch.setenv("DEV_MODE", "false")
    reset_settings_cache_for_tests()
    refresh_genai_clients_if_needed(error_app)

    client = TestClient(error_app, raise_server_exceptions=False)
    response = client.get("/dashboard/__test_error__")

    assert response.status_code == 500
    assert "Something went wrong" in response.text
    assert "Traceback" not in response.text
    assert "boom test error" not in response.text


def test_dashboard_http_exception_renders_html(error_app) -> None:
    client = TestClient(error_app, raise_server_exceptions=False)
    response = client.get("/dashboard/__test_http_error__")

    assert response.status_code == 404
    assert "Page not found" in response.text
    assert "Missing bot" in response.text


def test_api_error_stays_json_in_dev(error_app, monkeypatch) -> None:
    monkeypatch.setenv("DEV_MODE", "true")
    reset_settings_cache_for_tests()
    refresh_genai_clients_if_needed(error_app)

    client = TestClient(error_app, raise_server_exceptions=False)
    response = client.get("/api/__test_error__")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["detail"]["type"] == "ValueError"
    assert "api boom error" in payload["detail"]["message"]
