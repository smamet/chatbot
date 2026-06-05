from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from chatbot.adapters.channels import instagram_meta
from chatbot.adapters.channels.meta_signature import verify_signature
from chatbot.application.channel_outbound import (
    get_outbound_connector,
    queue_pending_reply,
    should_queue_for_validation,
)
from chatbot.application.chat_service import ChatService
from chatbot.application.connector_service import ConnectorService
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import ConnectorType
from chatbot.interfaces.api.deps import (
    get_connector_service,
    get_session,
    get_settings_dep,
    get_webhook_chat_service,
    get_webhook_tenant,
)

router = APIRouter()


def _ig_cfg(connectors: ConnectorService, tenant_id: int) -> dict:
    return connectors.get_instagram_config(tenant_id) or {}


@router.get("/webhooks/instagram/{slug}")
async def verify_instagram(
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
    cfg = _ig_cfg(connectors, tenant.id)
    expected = str(cfg.get("verify_token", "")).strip() or settings.instagram_effective_verify_token
    if hub_verify_token != expected:
        raise HTTPException(status_code=403, detail="invalid token")
    return Response(content=hub_challenge, media_type="text/plain")


@router.post("/webhooks/instagram/{slug}")
async def instagram_inbound(
    slug: str,
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    tenant=Depends(get_webhook_tenant),
    connectors: ConnectorService = Depends(get_connector_service),
    service: ChatService = Depends(get_webhook_chat_service),
    session: Session = Depends(get_session),
):
    raw = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    cfg = _ig_cfg(connectors, tenant.id)
    secret = str(cfg.get("app_secret", "")).strip() or settings.whatsapp_app_secret
    if secret and not verify_signature(raw, sig, secret):
        raise HTTPException(status_code=403, detail="bad signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json")
    ig_id, text = instagram_meta.extract_first_text_message(payload)
    if not ig_id or not text:
        return {"status": "ignored"}
    session_id = f"instagram:{ig_id}"
    result = service.handle_user_message(session_id, text)
    out_conn = get_outbound_connector(connectors, tenant.id, ConnectorType.INSTAGRAM)
    if should_queue_for_validation(out_conn):
        queue_pending_reply(
            session,
            tenant_id=tenant.id,
            connector_id=out_conn.id,
            session_id=session_id,
            channel=ConnectorType.INSTAGRAM.value,
            recipient_id=ig_id,
            draft_text=result.text,
        )
        return {"status": "queued"}
    token = str(cfg.get("access_token", "")).strip() or settings.instagram_access_token
    ig_user = str(cfg.get("ig_user_id", "")).strip() or settings.instagram_ig_user_id
    if token and ig_user:
        instagram_meta.send_instagram_text(
            access_token=token,
            ig_user_id=ig_user,
            recipient_igsid=ig_id,
            text=result.text,
        )
    return {"status": "ok"}
