from datetime import UTC, datetime
import time
from unittest.mock import patch

import pytest

from evenor.application.mail_oauth_service import (
    MailOAuthError,
    apply_oauth_tokens_to_config,
    get_oauth_access_token,
    is_oauth_connected,
    prepare_oauth_mail_config,
    resolve_mail_oauth_credentials,
)
from evenor.adapters.microsoft.oauth import OAuthTokens
from evenor.config.settings import Settings
from evenor.domain.models.mail_connection import MailConnection, MailConnectionProvider


def test_is_oauth_connected() -> None:
    assert not is_oauth_connected({"auth_type": "password"})
    assert is_oauth_connected(
        {"auth_type": "microsoft_oauth", "oauth_refresh_token": "rt"}
    )


def test_get_oauth_access_token_uses_cached_token() -> None:
    config = {
        "auth_type": "microsoft_oauth",
        "oauth_refresh_token": "rt",
        "oauth_access_token": "cached",
        "oauth_token_expires_at": int(time.time()) + 3600,
        "microsoft_client_id": "cid",
        "microsoft_client_secret": "sec",
    }
    result = get_oauth_access_token(config, direction="in")
    assert result.access_token == "cached"
    assert result.updated_config is None


@patch("evenor.application.mail_oauth_service.microsoft_oauth.refresh_access_token")
def test_get_oauth_access_token_refreshes_when_expired(mock_refresh) -> None:
    mock_refresh.return_value = OAuthTokens(
        access_token="new",
        refresh_token="rt2",
        expires_at=int(time.time()) + 3000,
    )
    config = {
        "auth_type": "microsoft_oauth",
        "oauth_refresh_token": "rt",
        "oauth_access_token": "old",
        "oauth_token_expires_at": int(time.time()) - 10,
        "microsoft_client_id": "cid",
        "microsoft_client_secret": "sec",
    }
    result = get_oauth_access_token(config, direction="in")
    assert result.access_token == "new"
    assert result.updated_config is not None
    assert result.updated_config["oauth_refresh_token"] == "rt2"


def test_prepare_oauth_mail_config_password_passthrough() -> None:
    cfg = {"auth_type": "password", "imap_host": "h"}
    mail_cfg, updated = prepare_oauth_mail_config(cfg, direction="in")
    assert mail_cfg == cfg
    assert updated is None


def test_apply_oauth_tokens_to_config() -> None:
    tokens = OAuthTokens(access_token="a", refresh_token="r", expires_at=123)
    out = apply_oauth_tokens_to_config({}, tokens)
    assert out["oauth_access_token"] == "a"
    assert out["oauth_refresh_token"] == "r"


def test_get_oauth_access_token_missing_refresh() -> None:
    with pytest.raises(MailOAuthError, match="not connected"):
        get_oauth_access_token({"auth_type": "google_oauth"}, direction="in")


def _sample_connection(**overrides) -> MailConnection:
    now = datetime.now(UTC)
    defaults = {
        "id": 1,
        "tenant_id": 1,
        "label": "x",
        "provider": MailConnectionProvider.MICROSOFT_OAUTH,
        "mailbox_email": "a@b.com",
        "config": {"microsoft_client_id": "row-cid", "microsoft_client_secret": "row-sec"},
        "active": True,
        "created_at": now,
        "updated_at": now,
    }
    defaults.update(overrides)
    return MailConnection(**defaults)


def test_resolve_mail_oauth_credentials_prefers_platform_env() -> None:
    settings = Settings(
        microsoft_mail_client_id="env-cid",
        microsoft_mail_client_secret="env-sec",
    )
    connection = _sample_connection()
    client_id, client_secret = resolve_mail_oauth_credentials(connection, settings)
    assert client_id == "env-cid"
    assert client_secret == "env-sec"


def test_resolve_mail_oauth_credentials_falls_back_to_connection() -> None:
    settings = Settings()
    connection = _sample_connection()
    client_id, client_secret = resolve_mail_oauth_credentials(connection, settings)
    assert client_id == "row-cid"
    assert client_secret == "row-sec"


@patch("evenor.application.mail_oauth_service.microsoft_oauth.refresh_access_token")
def test_get_oauth_access_token_uses_platform_env_on_refresh(mock_refresh) -> None:
    mock_refresh.return_value = OAuthTokens(
        access_token="new",
        refresh_token="rt2",
        expires_at=int(time.time()) + 3000,
    )
    settings = Settings(
        microsoft_mail_client_id="env-cid",
        microsoft_mail_client_secret="env-sec",
    )
    config = {
        "auth_type": "microsoft_oauth",
        "oauth_refresh_token": "rt",
        "oauth_access_token": "old",
        "oauth_token_expires_at": int(time.time()) - 10,
    }
    result = get_oauth_access_token(config, direction="in", settings=settings)
    assert result.access_token == "new"
    mock_refresh.assert_called_once_with(
        refresh_token="rt",
        client_id="env-cid",
        client_secret="env-sec",
    )
