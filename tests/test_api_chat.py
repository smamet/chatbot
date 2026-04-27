from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.interfaces.api.deps import get_chat_service
from chatbot.interfaces.api.main import create_app


def test_healthz() -> None:
    client = TestClient(create_app())
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.fixture
def chat_secret_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("CHAT_API_SECRET", "supersecret")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    reset_settings_cache_for_tests()
    app = create_app()

    class _FakeChatService:
        def handle_user_message(self, session_id: str, message: str):
            return SimpleNamespace(
                text="ok",
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    candidates_tokens=1,
                    total_tokens=2,
                ),
            )

    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()


def test_v1_chat_requires_bearer_when_secret_set(chat_secret_client: TestClient) -> None:
    r = chat_secret_client.post("/v1/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 401
    assert r.json().get("detail") == "Unauthorized"


def test_v1_chat_rejects_wrong_bearer(chat_secret_client: TestClient) -> None:
    r = chat_secret_client.post(
        "/v1/chat",
        json={"session_id": "s1", "message": "hi"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_v1_chat_accepts_bearer(chat_secret_client: TestClient) -> None:
    r = chat_secret_client.post(
        "/v1/chat",
        json={"session_id": "s1", "message": "hi"},
        headers={"Authorization": "Bearer supersecret"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "ok"
    assert body["usage"]["total_tokens"] == 2
