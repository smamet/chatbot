from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from datetime import UTC, datetime

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.config.settings import reset_settings_cache_for_tests
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.interfaces.api.deps import get_connector_service, get_webhook_chat_service, get_webhook_tenant
from chatbot.interfaces.api.main import create_app

WEBHOOK_SLUG = "meta-bot"


def _sig(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class _FakeConnectors:
    def get_messenger_config(self, tenant_id: int, *, outbound: bool = False) -> dict:
        _ = tenant_id, outbound
        return {"verify_token": "verify-shared", "app_secret": "meta-secret", "page_access_token": "page-token"}

    def get_instagram_config(self, tenant_id: int, *, outbound: bool = False) -> dict:
        _ = tenant_id, outbound
        return {
            "verify_token": "verify-shared",
            "app_secret": "meta-secret",
            "access_token": "ig-token",
            "ig_user_id": "IG_USER",
        }

    def find(
        self,
        tenant_id: int,
        *,
        direction: ConnectorDirection,
        type: ConnectorType,
    ) -> Connector:
        _ = tenant_id
        return Connector(
            id=1,
            tenant_id=1,
            direction=direction,
            type=type,
            mode=ConnectorMode.DIRECT,
            config={},
            active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )


@pytest.fixture
def meta_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'meta.db'}")
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-shared")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "meta-secret")
    reset_settings_cache_for_tests()
    from chatbot.config.settings import get_settings

    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    app = create_app()
    app.state.session_factory = session_factory(engine)
    fake_tenant = SimpleNamespace(id=1, slug=WEBHOOK_SLUG, active=True)

    class _FakeChatService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def handle_user_message(self, session_id: str, message: str):
            self.calls.append((session_id, message))
            return SimpleNamespace(
                text="ok",
                usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            )

    fake = _FakeChatService()
    app.dependency_overrides[get_webhook_chat_service] = lambda: fake
    app.dependency_overrides[get_webhook_tenant] = lambda slug=WEBHOOK_SLUG: fake_tenant
    app.dependency_overrides[get_connector_service] = lambda: _FakeConnectors()
    client = TestClient(app)
    client.fake_service = fake  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()
    engine.dispose()


def test_verify_messenger_uses_connector_token(meta_client: TestClient) -> None:
    r = meta_client.get(
        f"/webhooks/messenger/{WEBHOOK_SLUG}",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-shared", "hub.challenge": "42"},
    )
    assert r.status_code == 200
    assert r.text == "42"


def test_verify_instagram_uses_connector_token(meta_client: TestClient) -> None:
    r = meta_client.get(
        f"/webhooks/instagram/{WEBHOOK_SLUG}",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-shared", "hub.challenge": "43"},
    )
    assert r.status_code == 200
    assert r.text == "43"


def test_messenger_post_rejects_bad_signature(meta_client: TestClient) -> None:
    payload = b'{"object":"page","entry":[]}'
    r = meta_client.post(
        f"/webhooks/messenger/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert r.status_code == 403


def test_instagram_post_rejects_bad_signature(meta_client: TestClient) -> None:
    payload = b'{"object":"instagram","entry":[]}'
    r = meta_client.post(
        f"/webhooks/instagram/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert r.status_code == 403


def test_messenger_post_text_inbound_ok(meta_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chatbot.interfaces.api.routers.messenger_webhook.messenger_meta.send_messenger_text",
        lambda **kwargs: None,
    )
    body = {
        "object": "page",
        "entry": [
            {
                "id": "PAGE_ID",
                "time": 1,
                "messaging": [
                    {"sender": {"id": "PSID_1"}, "recipient": {"id": "PAGE_ID"}, "message": {"text": "hello"}}
                ],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    r = meta_client.post(
        f"/webhooks/messenger/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("meta-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert meta_client.fake_service.calls == [("messenger:PSID_1", "hello")]  # type: ignore[attr-defined]


def test_instagram_post_text_inbound_ok(meta_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chatbot.interfaces.api.routers.instagram_webhook.instagram_meta.send_instagram_text",
        lambda **kwargs: None,
    )
    body = {
        "object": "instagram",
        "entry": [
            {
                "id": "IG_USER",
                "time": 1,
                "messaging": [
                    {
                        "sender": {"id": "IGSID_1"},
                        "recipient": {"id": "IG_USER"},
                        "message": {"text": "hello ig"},
                    }
                ],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    r = meta_client.post(
        f"/webhooks/instagram/{WEBHOOK_SLUG}",
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
                "messaging": [
                    {
                        "sender": {"id": "PSID_1"},
                        "recipient": {"id": "PAGE_ID"},
                        "message": {"is_echo": True, "text": "echo"},
                    }
                ],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    before = len(meta_client.fake_service.calls)  # type: ignore[attr-defined]
    r = meta_client.post(
        f"/webhooks/messenger/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("meta-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ignored"
    assert len(meta_client.fake_service.calls) == before  # type: ignore[attr-defined]
