from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evenor.adapters.google.oauth import (
    OAuthTokens,
    build_authorize_url,
)
from evenor.adapters.microsoft.oauth import (
    OAuthTokens as MsOAuthTokens,
    build_authorize_url as ms_build_authorize_url,
    exchange_code as ms_exchange_code,
    scopes_for_connection as ms_scopes_for_connection,
    scopes_for_direction as ms_scopes_for_direction,
)


def test_microsoft_scopes_for_direction() -> None:
    assert "IMAP.AccessAsUser.All" in " ".join(ms_scopes_for_direction("in"))
    assert "SMTP.Send" in " ".join(ms_scopes_for_direction("out"))


def test_microsoft_scopes_for_connection_includes_imap_and_smtp() -> None:
    scopes = " ".join(ms_scopes_for_connection())
    assert "IMAP.AccessAsUser.All" in scopes
    assert "SMTP.Send" in scopes


def test_microsoft_authorize_url_for_connection() -> None:
    url = ms_build_authorize_url(
        client_id="cid",
        redirect_uri="https://app/cb",
        state="state123",
        for_connection=True,
    )
    assert "IMAP.AccessAsUser.All" in url
    assert "SMTP.Send" in url


def test_microsoft_authorize_url_contains_client() -> None:
    url = ms_build_authorize_url(
        client_id="cid",
        redirect_uri="https://app/cb",
        state="state123",
        direction="in",
    )
    assert "client_id=cid" in url
    assert "state=state123" in url


def test_google_authorize_url_offline_consent() -> None:
    url = build_authorize_url(
        client_id="gid",
        redirect_uri="https://app/cb",
        state="st",
        direction="in",
    )
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "mail.google.com" in url


@patch("evenor.adapters.microsoft.oauth.httpx.Client")
def test_microsoft_exchange_code(mock_client_cls) -> None:
    response = MagicMock()
    response.json.return_value = {
        "access_token": "at",
        "refresh_token": "rt",
        "expires_in": 3600,
    }
    mock_client_cls.return_value.__enter__.return_value.post.return_value = response

    tokens = ms_exchange_code(
        code="code",
        client_id="cid",
        client_secret="sec",
        redirect_uri="https://app/cb",
    )
    assert isinstance(tokens, MsOAuthTokens)
    assert tokens.access_token == "at"
    assert tokens.refresh_token == "rt"


@patch("evenor.adapters.google.oauth.httpx.Client")
def test_google_refresh_access_token(mock_client_cls) -> None:
    response = MagicMock()
    response.json.return_value = {
        "access_token": "new-at",
        "expires_in": 3600,
    }
    mock_client_cls.return_value.__enter__.return_value.post.return_value = response

    from evenor.adapters.google.oauth import refresh_access_token as google_refresh

    tokens = google_refresh(
        refresh_token="rt",
        client_id="gid",
        client_secret="gsec",
    )
    assert tokens.access_token == "new-at"
    assert tokens.refresh_token == "rt"
