from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from chatbot.adapters.channels import instagram_meta
from chatbot.adapters.channels.meta_signature import verify_signature
from chatbot.application.chat_service import ChatService
from chatbot.config.settings import Settings
from chatbot.interfaces.api.deps import get_chat_service, get_settings_dep

router = APIRouter()


@router.get("/webhooks/instagram")
async def verify_instagram(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    settings: Settings = Depends(get_settings_dep),
):
    if hub_mode != "subscribe" or not hub_challenge:
        raise HTTPException(status_code=403, detail="invalid")
    if hub_verify_token != settings.instagram_effective_verify_token:
        raise HTTPException(status_code=403, detail="invalid token")
    return Response(content=hub_challenge, media_type="text/plain")


@router.post("/webhooks/instagram")
async def instagram_inbound(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    service: ChatService = Depends(get_chat_service),
):
    raw = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    if settings.whatsapp_app_secret and not verify_signature(raw, sig, settings.whatsapp_app_secret):
        raise HTTPException(status_code=403, detail="bad signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json")
    igsid, text = instagram_meta.extract_first_text_message(payload)
    if not igsid or not text:
        return {"status": "ignored"}
    session_id = f"instagram:{igsid}"
    result = service.handle_user_message(session_id, text)
    if settings.instagram_ig_user_id and settings.instagram_access_token:
        instagram_meta.send_instagram_text(
            ig_user_id=settings.instagram_ig_user_id,
            access_token=settings.instagram_access_token,
            recipient_igsid=igsid,
            text=result.text,
        )
    return {"status": "ok"}
