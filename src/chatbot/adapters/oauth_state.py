from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any


def sign_mail_connection_oauth_state(
    *,
    slug: str,
    connection_id: int,
    provider: str,
    secret: str,
) -> str:
    payload = json.dumps(
        {
            "slug": slug,
            "connection_id": connection_id,
            "provider": provider,
            "ts": int(time.time()),
        },
        separators=(",", ":"),
    )
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_mail_connection_oauth_state(
    state: str,
    *,
    secret: str,
    max_age_seconds: int = 600,
) -> dict[str, str | int]:
    if "." not in state:
        raise ValueError("Invalid OAuth state")
    payload_str, sig = state.rsplit(".", 1)
    expected = hmac.new(
        secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise ValueError("Invalid OAuth state signature")
    payload: dict[str, Any] = json.loads(payload_str)
    slug = str(payload.get("slug", "")).strip()
    provider = str(payload.get("provider", "")).strip()
    connection_id = int(payload.get("connection_id", 0))
    ts = int(payload.get("ts", 0))
    if not slug or not provider or connection_id <= 0:
        raise ValueError("Invalid OAuth state payload")
    if int(time.time()) - ts > max_age_seconds:
        raise ValueError("Expired OAuth state")
    return {"slug": slug, "connection_id": connection_id, "provider": provider}


def sign_connector_oauth_state(
    *,
    slug: str,
    direction: str,
    provider: str,
    secret: str,
) -> str:
    payload = json.dumps(
        {"slug": slug, "direction": direction, "provider": provider, "ts": int(time.time())},
        separators=(",", ":"),
    )
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_connector_oauth_state(
    state: str,
    *,
    secret: str,
    max_age_seconds: int = 600,
) -> dict[str, str]:
    if "." not in state:
        raise ValueError("Invalid OAuth state")
    payload_str, sig = state.rsplit(".", 1)
    expected = hmac.new(
        secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise ValueError("Invalid OAuth state signature")
    payload: dict[str, Any] = json.loads(payload_str)
    slug = str(payload.get("slug", "")).strip()
    direction = str(payload.get("direction", "")).strip()
    provider = str(payload.get("provider", "")).strip()
    ts = int(payload.get("ts", 0))
    if not slug or not direction or not provider:
        raise ValueError("Invalid OAuth state payload")
    if int(time.time()) - ts > max_age_seconds:
        raise ValueError("Expired OAuth state")
    return {"slug": slug, "direction": direction, "provider": provider}
