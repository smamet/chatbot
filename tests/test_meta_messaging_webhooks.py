from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.interfaces.api.deps import get_chat_service
from chatbot.interfaces.api.main import create_app


def _sig(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


@pytest.fixture
def meta_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'meta.db'}")
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-shared")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "meta-secret")
    monkeypatch.setenv("MESSENGER_PAGE_ACCESS_TOKEN", "")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "")
    monkeypatch.setenv("INSTAGRAM_IG_USER_ID", "")
    reset_settings_cache_for_tests()
    app = create_app()

    class _FakeChatService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def handle_user_message(self, session_id: str, message: str):
            self.calls.append((session_id, message))
            return SimpleNamespace(text="ok", usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2))

    fake = _FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: fake
    client = TestClient(app)
    client.fake_service = fake  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()


def test_verify_messenger_uses_whatsapp_token_fallback(meta_client: TestClient) -> None:
    r = meta_client.get(
        "/webhooks/messenger",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-shared", "hub.challenge": "42"},
    )
    assert r.status_code == 200
    assert r.text == "42"


def test_verify_instagram_uses_whatsapp_token_fallback(meta_client: TestClient) -> None:
    r = meta_client.get(
        "/webhooks/instagram",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-shared", "hub.challenge": "43"},
    )
    assert r.status_code == 200
    assert r.text == "43"


def test_messenger_post_rejects_bad_signature(meta_client: TestClient) -> None:
    payload = b'{"object":"page","entry":[]}'
    r = meta_client.post(
        "/webhooks/messenger",
        content=payload,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert r.status_code == 403


def test_instagram_post_rejects_bad_signature(meta_client: TestClient) -> None:
    payload = b'{"object":"instagram","entry":[]}'
    r = meta_client.post(
        "/webhooks/instagram",
        content=payload,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert r.status_code == 403


def test_messenger_post_text_inbound_ok(meta_client: TestClient) -> None:
    body = {
        "object": "page",
        "entry": [
            {
                "id": "PAGE_ID",
                "time": 1,
                "messaging": [{"sender": {"id": "PSID_1"}, "recipient": {"id": "PAGE_ID"}, "message": {"text": "hello"}}],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    r = meta_client.post(
        "/webhooks/messenger",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("meta-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert meta_client.fake_service.calls == [("messenger:PSID_1", "hello")]  # type: ignore[attr-defined]


def test_instagram_post_text_inbound_ok(meta_client: TestClient) -> None:
    body = {
        "object": "instagram",
        "entry": [
            {
                "id": "IG_USER",
                "time": 1,
                "messaging": [{"sender": {"id": "IGSID_1"}, "recipient": {"id": "IG_USER"}, "message": {"text": "hello ig"}}],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    r = meta_client.post(
        "/webhooks/instagram",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("meta-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert meta_client.fake_service.calls[-1] == ("instagram:IGSID_1", "hello ig")  # type: ignore[attr-defined]


def test_messenger_echo_message_is_ignored(meta_client: TestClient) -> None:
    body = {
        "object": "page",
        "entry": [
            {
                "id": "PAGE_ID",
                "time": 1,
                "messaging": [{"sender": {"id": "PSID_1"}, "recipient": {"id": "PAGE_ID"}, "message": {"is_echo": True, "text": "echo"}}],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    before = len(meta_client.fake_service.calls)  # type: ignore[attr-defined]
    r = meta_client.post(
        "/webhooks/messenger",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("meta-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ignored"
    assert len(meta_client.fake_service.calls) == before  # type: ignore[attr-defined]
