from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
MAIL_SCOPE = "https://mail.google.com/"


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: int


def scopes_for_direction(_direction: str) -> tuple[str, ...]:
    return (MAIL_SCOPE,)


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    direction: str,
) -> str:
    scopes = " ".join(scopes_for_direction(direction))
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
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
