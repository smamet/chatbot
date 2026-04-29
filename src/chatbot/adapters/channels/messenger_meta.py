from __future__ import annotations

import json
from typing import Any

import httpx
from chatbot.adapters.channels.text_format import format_for_messenger

GRAPH_API_VERSION = "v21.0"


def extract_first_text_message(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (psid, text) from Messenger webhook payload if present."""
    try:
        if payload.get("object") != "page":
            return None, None
        entries = payload.get("entry") or []
        for entry in entries:
            messaging = entry.get("messaging") or []
            for event in messaging:
                message = event.get("message") or {}
                if message.get("is_echo"):
                    continue
                text = message.get("text")
                sender_id = (event.get("sender") or {}).get("id")
                if sender_id and isinstance(text, str) and text.strip():
                    return str(sender_id), text
    except (TypeError, KeyError, AttributeError):
        pass
    return None, None


def send_messenger_text(
    *,
    page_access_token: str,
    recipient_psid: str,
    text: str,
    timeout: float = 30.0,
) -> None:
    body_text = format_for_messenger(text)
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/me/messages"
    headers = {
        "Authorization": f"Bearer {page_access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_type": "RESPONSE",
        "recipient": {"id": recipient_psid},
        "message": {"text": body_text[:2000]},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
