from __future__ import annotations

import json
from typing import Any

import httpx
from evenor.adapters.channels.text_format import format_for_instagram

GRAPH_API_VERSION = "v25.0"


def extract_first_text_message(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (igsid, text) from Instagram messaging webhook payload if present."""
    try:
        if payload.get("object") != "instagram":
            return None, None
        entries = payload.get("entry") or []
        for entry in entries:
            messaging = entry.get("messaging") or []
            for event in messaging:
                message = event.get("message") or {}
                if message.get("is_echo") or message.get("is_deleted"):
                    continue
                text = message.get("text")
                sender_id = (event.get("sender") or {}).get("id")
                if sender_id and isinstance(text, str) and text.strip():
                    return str(sender_id), text
    except (TypeError, KeyError, AttributeError):
        pass
    return None, None


def send_instagram_text(
    *,
    ig_user_id: str,
    access_token: str,
    recipient_igsid: str,
    text: str,
    timeout: float = 30.0,
) -> None:
    body_text = format_for_instagram(text)
    url = f"https://graph.instagram.com/{GRAPH_API_VERSION}/{ig_user_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "recipient": {"id": recipient_igsid},
        "message": {"text": body_text[:1000]},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
