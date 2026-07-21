from __future__ import annotations

from unittest.mock import MagicMock, patch

from chatbot.adapters.quickbooks.oauth import (
    OAuthTokens,
    build_authorize_url,
    exchange_code,
    sign_oauth_state,
    verify_oauth_state,
)


def test_build_authorize_url_contains_client_id() -> None:
    url = build_authorize_url(
        client_id="cid",
        redirect_uri="https://app.example/callback",
        state="signed-state",
        environment="sandbox",
    )
    assert "client_id=cid" in url
    assert "signed-state" in url


def test_sign_and_verify_oauth_state() -> None:
    state = sign_oauth_state(slug="my-bot", secret="test-secret")
    assert verify_oauth_state(state, secret="test-secret") == "my-bot"


def test_exchange_code_parses_tokens() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
            }

    with patch("chatbot.adapters.quickbooks.oauth.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = FakeResponse()
        tokens = exchange_code(
            code="abc",
            client_id="cid",
            client_secret="sec",
            redirect_uri="https://app/cb",
        )
    assert isinstance(tokens, OAuthTokens)
    assert tokens.access_token == "access"
    assert tokens.refresh_token == "refresh"
