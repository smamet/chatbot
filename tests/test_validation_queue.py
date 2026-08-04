from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.connector_service import ConnectorService
from evenor.application.tenant_service import TenantService
from evenor.config.settings import reset_settings_cache_for_tests
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from evenor.domain.models.pending_reply import PendingReplyStatus
from evenor.interfaces.api.deps import get_webhook_chat_service, get_webhook_tenant
from evenor.interfaces.api.main import create_app

WEBHOOK_SLUG = "validation-bot"


def _sig(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _wa_payload(wa_id: str = "23057770000", text: str = "hello") -> bytes:
    body = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"type": "text", "from": wa_id, "text": {"body": text}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    return json.dumps(body).encode("utf-8")


@pytest.fixture
def validation_webhook_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    db = tmp_path / "validation.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LANCEDB_ROOT", str(tmp_path / "lancedb"))
    reset_settings_cache_for_tests()
    from evenor.config.settings import get_settings

    settings = get_settings()
    engine = create_db_engine(settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        result = tenant_svc.create_tenant(name="Validation Bot", slug=WEBHOOK_SLUG)
        conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        conn_svc.upsert(
            tenant_id=result.tenant.id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.DIRECT,
            config={"verify_token": "verify-wa", "app_secret": "wa-secret"},
        )
        conn_svc.upsert(
            tenant_id=result.tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.VALIDATION,
            config={
                "phone_number_id": "phone-id",
                "access_token": "access-token",
            },
        )
        session.commit()
        tenant_id = result.tenant.id

    app = create_app()
    app.state.session_factory = factory
    fake_tenant = SimpleNamespace(id=tenant_id, slug=WEBHOOK_SLUG, active=True)

    class _FakeChatService:
        def handle_user_message(self, session_id: str, message: str):
            _ = session_id, message
            return SimpleNamespace(
                text="draft reply",
                usage=SimpleNamespace(prompt_tokens=1, candidates_tokens=1, total_tokens=2),
            )

    app.dependency_overrides[get_webhook_chat_service] = lambda: _FakeChatService()
    app.dependency_overrides[get_webhook_tenant] = lambda slug=WEBHOOK_SLUG: fake_tenant

    with TestClient(app) as client:
        yield client, factory, tenant_id
    app.dependency_overrides.clear()
    reset_settings_cache_for_tests()
    engine.dispose()


def test_whatsapp_validation_mode_queues_reply(validation_webhook_env, monkeypatch) -> None:
    client, factory, tenant_id = validation_webhook_env
    sent: list[str] = []
    monkeypatch.setattr(
        "evenor.interfaces.api.routers.whatsapp_webhook.whatsapp_meta.send_whatsapp_text",
        lambda **kwargs: sent.append(kwargs["text"]),
    )
    payload = _wa_payload()
    r = client.post(
        f"/webhooks/whatsapp/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("wa-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert sent == []
    with factory() as session:
        pending = SqlAlchemyPendingReplyRepository(session).list_pending(tenant_id)
    assert len(pending) == 1
    assert pending[0].draft_text == "draft reply"
    assert pending[0].recipient_id == "23057770000"


def test_whatsapp_direct_mode_sends_reply(validation_webhook_env, monkeypatch) -> None:
    client, factory, tenant_id = validation_webhook_env
    with factory() as session:
        conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        out = conn_svc.find(tenant_id, direction=ConnectorDirection.OUT, type=ConnectorType.WHATSAPP)
        assert out is not None
        conn_svc.upsert(
            tenant_id=tenant_id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.WHATSAPP,
            mode=ConnectorMode.DIRECT,
            config=out.config,
        )
        session.commit()
    sent: list[str] = []
    monkeypatch.setattr(
        "evenor.interfaces.api.routers.whatsapp_webhook.whatsapp_meta.send_whatsapp_text",
        lambda **kwargs: sent.append(kwargs["text"]),
    )
    payload = _wa_payload(text="direct")
    r = client.post(
        f"/webhooks/whatsapp/{WEBHOOK_SLUG}",
        content=payload,
        headers={"X-Hub-Signature-256": _sig("wa-secret", payload)},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert sent == ["draft reply"]
