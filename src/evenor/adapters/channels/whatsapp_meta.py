from __future__ import annotations

import json
from typing import Any

import httpx
from evenor.adapters.channels.meta_signature import verify_signature
from evenor.adapters.channels.text_format import format_for_whatsapp


def extract_first_text_message(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (wa_user_id, text) from WhatsApp Cloud webhook payload if present."""
    try:
        entries = payload.get("entry") or []
        for entry in entries:
            changes = entry.get("changes") or []
            for change in changes:
                value = change.get("value") or {}
                messages = value.get("messages") or []
                for msg in messages:
                    if msg.get("type") != "text":
                        continue
                    from_id = msg.get("from")
                    body = (msg.get("text") or {}).get("body")
                    if from_id and body:
                        return str(from_id), str(body)
    except (TypeError, KeyError, AttributeError):
        pass
    return None, None


def send_whatsapp_text(
    *,
    phone_number_id: str,
    access_token: str,
    to_wa_id: str,
    text: str,
    timeout: float = 30.0,
) -> None:
    body_text = format_for_whatsapp(text)
    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "to": to_wa_id,
        "type": "text",
        "text": {"preview_url": False, "body": body_text[:4096]},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()


def upload_media(
    *,
    phone_number_id: str,
    access_token: str,
    data: bytes,
    mime_type: str,
    timeout: float = 60.0,
) -> str:
    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/media"
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"file": ("attachment", data, mime_type)}
    form = {"messaging_product": "whatsapp", "type": mime_type}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, data=form, files=files)
        r.raise_for_status()
        payload = r.json()
    media_id = payload.get("id")
    if not media_id:
        raise RuntimeError("WhatsApp media upload did not return id")
    return str(media_id)


def send_whatsapp_document(
    *,
    phone_number_id: str,
    access_token: str,
    to_wa_id: str,
    media_id: str,
    filename: str,
    caption: str | None = None,
    timeout: float = 30.0,
) -> None:
    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    document: dict[str, str] = {"id": media_id, "filename": filename}
    if caption:
        document["caption"] = format_for_whatsapp(caption)[:1024]
    body = {
        "messaging_product": "whatsapp",
        "to": to_wa_id,
        "type": "document",
        "document": document,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
