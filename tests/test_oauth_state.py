from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from evenor.adapters.oauth_state import (
    sign_connector_oauth_state,
    sign_mail_connection_oauth_state,
    verify_connector_oauth_state,
    verify_mail_connection_oauth_state,
)


def test_sign_and_verify_mail_connection_oauth_state() -> None:
    state = sign_mail_connection_oauth_state(
        slug="my-bot",
        connection_id=42,
        provider="microsoft",
        secret="test-secret",
    )
    data = verify_mail_connection_oauth_state(state, secret="test-secret")
    assert data["slug"] == "my-bot"
    assert data["connection_id"] == 42
    assert data["provider"] == "microsoft"


def test_sign_and_verify_connector_oauth_state() -> None:
    state = sign_connector_oauth_state(
        slug="my-bot",
        direction="in",
        provider="microsoft",
        secret="test-secret",
    )
    data = verify_connector_oauth_state(state, secret="test-secret")
    assert data == {"slug": "my-bot", "direction": "in", "provider": "microsoft"}


def test_verify_connector_oauth_state_rejects_tampered() -> None:
    state = sign_connector_oauth_state(
        slug="my-bot",
        direction="out",
        provider="google",
        secret="test-secret",
    )
    with pytest.raises(ValueError, match="signature"):
        verify_connector_oauth_state(state + "x", secret="test-secret")


def test_verify_connector_oauth_state_expired() -> None:
    with patch("evenor.adapters.oauth_state.time") as mock_time:
        mock_time.time.return_value = 1000
        state = sign_connector_oauth_state(
            slug="bot",
            direction="in",
            provider="microsoft",
            secret="secret",
        )
        mock_time.time.return_value = 1000 + 700
        with pytest.raises(ValueError, match="Expired"):
            verify_connector_oauth_state(state, secret="secret", max_age_seconds=600)
