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
def whatsapp_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'wa.db'}")
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-wa")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "wa-secret")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "phone-id")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "access-token")
    reset_settings_cache_for_tests()
    app = create_app()

    class _FakeChatService:
        def handle_user_message(self, session_id: str, message: str):
            _ = session_id
            _ = message
            return SimpleNamespace(text="clean reply", usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2))

    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()


def test_whatsapp_post_sends_clean_reply(monkeypatch: pytest.MonkeyPatch, whatsapp_client: TestClient) -> None:
    sent: list[str] = []

    def _fake_send_whatsapp_text(*, phone_number_id: str, access_token: str, to_wa_id: str, text: str, timeout=30.0):
        _ = phone_number_id
        _ = access_token
        _ = to_wa_id
        _ = timeout
        sent.append(text)

    monkeypatch.setattr("chatbot.interfaces.api.routers.whatsapp_webhook.whatsapp_meta.send_whatsapp_text", _fake_send_whatsapp_text)
    body = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "type": "text",
                                    "from": "23057770000",
                                    "text": {"body": "hello"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    payload = json.dumps(body).encode("utf-8")
    r = whatsapp_client.post(
        "/webhooks/whatsapp",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("wa-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert sent == ["clean reply"]
