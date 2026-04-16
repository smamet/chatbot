from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx


def verify_signature(raw_body: bytes, signature_header: str | None, app_secret: str) -> bool:
    if not signature_header or not app_secret:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = signature_header.removeprefix("sha256=")
    mac = hmac.new(app_secret.encode("utf-8"), msg=raw_body, digestmod=hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, expected)


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
    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "to": to_wa_id,
        "type": "text",
        "text": {"preview_url": False, "body": text[:4096]},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
