from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.interfaces.api.deps import get_connector_service, get_webhook_chat_service, get_webhook_tenant
from chatbot.interfaces.api.main import create_app

WEBHOOK_SLUG = "wa-bot"


def _sig(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class _FakeConnectors:
    def get_whatsapp_config(self, tenant_id: int, *, outbound: bool = False) -> dict:
        _ = tenant_id
        return {
            "verify_token": "verify-wa",
            "app_secret": "wa-secret",
            "phone_number_id": "phone-id",
            "access_token": "access-token",
        }


@pytest.fixture
def whatsapp_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'wa.db'}")
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    reset_settings_cache_for_tests()
    app = create_app()
    fake_tenant = SimpleNamespace(id=1, slug=WEBHOOK_SLUG, active=True)

    class _FakeChatService:
        def handle_user_message(self, session_id: str, message: str):
            _ = session_id
            _ = message
            return SimpleNamespace(
                text="clean reply",
                usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            )

    app.dependency_overrides[get_webhook_chat_service] = lambda: _FakeChatService()
    app.dependency_overrides[get_webhook_tenant] = lambda slug=WEBHOOK_SLUG: fake_tenant
    app.dependency_overrides[get_connector_service] = lambda: _FakeConnectors()
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

    monkeypatch.setattr(
        "chatbot.interfaces.api.routers.whatsapp_webhook.whatsapp_meta.send_whatsapp_text",
        _fake_send_whatsapp_text,
    )
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
        f"/webhooks/whatsapp/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("wa-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert sent == ["clean reply"]
