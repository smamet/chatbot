from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from chatbot.adapters.channels import whatsapp_meta
from chatbot.application.channel_outbound import (
    get_outbound_connector,
    queue_pending_reply,
    should_queue_for_validation,
)
from chatbot.application.chat_service import ChatService
from chatbot.application.connector_service import ConnectorService
from chatbot.domain.models.connector import ConnectorType
from chatbot.interfaces.api.deps import (
    get_connector_service,
    get_session,
    get_settings_dep,
    get_webhook_chat_service,
    get_webhook_tenant,
)
from chatbot.config.settings import Settings

router = APIRouter()


def _wa_cfg(connectors: ConnectorService, tenant_id: int) -> dict:
    cfg = connectors.get_whatsapp_config(tenant_id, outbound=False)
    return cfg or {}


@router.get("/webhooks/whatsapp/{slug}")
async def verify_whatsapp(
    slug: str,
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    tenant=Depends(get_webhook_tenant),
    connectors: ConnectorService = Depends(get_connector_service),
):
    _ = tenant
    if hub_mode != "subscribe" or not hub_challenge:
        raise HTTPException(status_code=403, detail="invalid")
    cfg = _wa_cfg(connectors, tenant.id)
    expected = str(cfg.get("verify_token", "")).strip()
    if not expected or hub_verify_token != expected:
        raise HTTPException(status_code=403, detail="invalid token")
    return Response(content=hub_challenge, media_type="text/plain")


@router.post("/webhooks/whatsapp/{slug}")
async def whatsapp_inbound(
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
    cfg = _wa_cfg(connectors, tenant.id)
    app_secret = str(cfg.get("app_secret", "")).strip() or settings.whatsapp_app_secret
    if app_secret and not whatsapp_meta.verify_signature(raw, sig, app_secret):
        raise HTTPException(status_code=403, detail="bad signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json")
    wa_id, text = whatsapp_meta.extract_first_text_message(payload)
    if not wa_id or not text:
        return {"status": "ignored"}
    session_id = f"whatsapp:{wa_id}"
    result = service.handle_user_message(session_id, text)
    out_conn = get_outbound_connector(connectors, tenant.id, ConnectorType.WHATSAPP)
    if should_queue_for_validation(out_conn):
        queue_pending_reply(
            session,
            tenant_id=tenant.id,
            connector_id=out_conn.id,
            session_id=session_id,
            channel=ConnectorType.WHATSAPP.value,
            recipient_id=wa_id,
            draft_text=result.text,
        )
        return {"status": "queued"}
    out_cfg = connectors.get_whatsapp_config(tenant.id, outbound=True) or cfg
    phone_id = str(out_cfg.get("phone_number_id", "")).strip()
    token = str(out_cfg.get("access_token", "")).strip()
    if phone_id and token:
        whatsapp_meta.send_whatsapp_text(
            phone_number_id=phone_id,
            access_token=token,
            to_wa_id=wa_id,
            text=result.text,
        )
    return {"status": "ok"}
