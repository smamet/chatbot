from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from chatbot.adapters.channels import whatsapp_meta
from chatbot.application.chat_service import ChatService
from chatbot.application.connector_service import ConnectorService
from chatbot.application.outbound_orchestrator import get_outbound_connector_for_channel, queue_after_chat
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
    out_conn = get_outbound_connector_for_channel(connectors, tenant.id, ConnectorType.WHATSAPP)
    if out_conn is None:
        return {"status": "ok"}
    status, _pending = queue_after_chat(
        session,
        tenant_id=tenant.id,
        connector=out_conn,
        session_id=session_id,
        recipient_id=wa_id,
        result=result,
        settings=settings,
        tenant_slug=tenant.slug,
    )
    session.commit()
    return {"status": status}
