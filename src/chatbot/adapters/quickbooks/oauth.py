from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

AUTHORIZE_URL = "https://appcenter.intuit.com/connect/oauth2"
TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
SCOPES = "com.intuit.quickbooks.accounting"


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: int
    realm_id: str | None = None


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    environment: str = "sandbox",
) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> OAuthTokens:
    return _token_request(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        client_id=client_id,
        client_secret=client_secret,
    )


def refresh_access_token(
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str,
) -> OAuthTokens:
    return _token_request(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        client_id=client_id,
        client_secret=client_secret,
        existing_refresh_token=refresh_token,
    )


def _token_request(
    data: dict[str, str],
    *,
    client_id: str,
    client_secret: str,
    existing_refresh_token: str | None = None,
) -> OAuthTokens:
    with httpx.Client(timeout=20.0) as client:
        response = client.post(
            TOKEN_URL,
            data=data,
            auth=(client_id, client_secret),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid token response")
    access = str(payload.get("access_token", "")).strip()
    refresh = str(payload.get("refresh_token", existing_refresh_token or "")).strip()
    expires_in = int(payload.get("expires_in", 3600))
    if not access or not refresh:
        raise ValueError("Missing tokens in OAuth response")
    return OAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at=int(time.time()) + max(60, expires_in - 60),
    )


def sign_oauth_state(*, slug: str, secret: str) -> str:
    payload = json.dumps({"slug": slug, "ts": int(time.time())}, separators=(",", ":"))
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_oauth_state(state: str, *, secret: str, max_age_seconds: int = 600) -> str:
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
    ts = int(payload.get("ts", 0))
    if not slug or int(time.time()) - ts > max_age_seconds:
        raise ValueError("Expired OAuth state")
    return slug
