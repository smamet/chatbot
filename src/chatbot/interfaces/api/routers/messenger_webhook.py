from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from chatbot.adapters.channels import messenger_meta
from chatbot.adapters.channels.meta_signature import verify_signature
from chatbot.application.chat_service import ChatService
from chatbot.application.connector_service import ConnectorService
from chatbot.config.settings import Settings
from chatbot.interfaces.api.deps import (
    get_connector_service,
    get_settings_dep,
    get_webhook_chat_service,
    get_webhook_tenant,
)

router = APIRouter()


def _msg_cfg(connectors: ConnectorService, tenant_id: int) -> dict:
    return connectors.get_messenger_config(tenant_id) or {}


@router.get("/webhooks/messenger/{slug}")
async def verify_messenger(
    slug: str,
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    tenant=Depends(get_webhook_tenant),
    connectors: ConnectorService = Depends(get_connector_service),
    settings: Settings = Depends(get_settings_dep),
):
    _ = tenant
    if hub_mode != "subscribe" or not hub_challenge:
        raise HTTPException(status_code=403, detail="invalid")
    cfg = _msg_cfg(connectors, tenant.id)
    expected = str(cfg.get("verify_token", "")).strip() or settings.messenger_effective_verify_token
    if hub_verify_token != expected:
        raise HTTPException(status_code=403, detail="invalid token")
    return Response(content=hub_challenge, media_type="text/plain")


@router.post("/webhooks/messenger/{slug}")
async def messenger_inbound(
    slug: str,
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    tenant=Depends(get_webhook_tenant),
    connectors: ConnectorService = Depends(get_connector_service),
    service: ChatService = Depends(get_webhook_chat_service),
):
    raw = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    cfg = _msg_cfg(connectors, tenant.id)
    secret = str(cfg.get("app_secret", "")).strip() or settings.whatsapp_app_secret
    if secret and not verify_signature(raw, sig, secret):
        raise HTTPException(status_code=403, detail="bad signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json")
    psid, text = messenger_meta.extract_first_text_message(payload)
    if not psid or not text:
        return {"status": "ignored"}
    result = service.handle_user_message(f"messenger:{psid}", text)
    token = str(cfg.get("page_access_token", "")).strip() or settings.messenger_page_access_token
    if token:
        messenger_meta.send_messenger_text(page_access_token=token, recipient_psid=psid, text=result.text)
    return {"status": "ok"}
