from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chatbot.application.connector_test_service import run_connector_connection_test


@patch.object(
    __import__("chatbot.adapters.mail.imap_client", fromlist=["ImapMailClient"]).ImapMailClient,
    "connect",
)
@patch.object(
    __import__("chatbot.adapters.mail.imap_client", fromlist=["ImapMailClient"]).ImapMailClient,
    "close",
)
def test_connector_test_imap_password(mock_close, mock_connect) -> None:
    result = run_connector_connection_test(
        "email",
        "in",
        {
            "auth_type": "password",
            "imap_host": "h",
            "username": "u",
            "password": "p",
        },
    )
    assert result.ok is True
    assert "IMAP" in result.message


@patch("chatbot.application.connector_test_service.prepare_oauth_mail_config")
@patch.object(
    __import__("chatbot.adapters.mail.imap_client", fromlist=["ImapMailClient"]).ImapMailClient,
    "connect",
)
@patch.object(
    __import__("chatbot.adapters.mail.imap_client", fromlist=["ImapMailClient"]).ImapMailClient,
    "close",
)
def test_connector_test_imap_oauth(mock_close, mock_connect, mock_prepare) -> None:
    mock_prepare.return_value = (
        {"auth_type": "microsoft_oauth", "_resolved_access_token": "tok", "username": "u"},
        None,
    )
    result = run_connector_connection_test(
        "email",
        "in",
        {"auth_type": "microsoft_oauth", "oauth_refresh_token": "rt"},
    )
    assert result.ok is True
    assert "OAuth" in result.message


@patch("chatbot.adapters.mail.smtp_sender.SmtpEmailSender.verify_connection")
def test_connector_test_smtp(mock_verify) -> None:
    result = run_connector_connection_test(
        "email",
        "out",
        {
            "auth_type": "password",
            "outbound_provider": "smtp",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
        },
    )
    assert result.ok is True
    mock_verify.assert_called_once()


@patch("chatbot.adapters.mail.mailjet_sender.httpx.Client")
def test_connector_test_mailjet(mock_client_cls) -> None:
    response = MagicMock()
    mock_client_cls.return_value.__enter__.return_value.get.return_value = response
    result = run_connector_connection_test(
        "email",
        "out",
        {
            "outbound_provider": "mailjet",
            "mailjet_api_key": "k",
            "mailjet_api_secret": "s",
        },
    )
    assert result.ok is True


def test_connector_test_unsupported_type() -> None:
    result = run_connector_connection_test("whatsapp", "in", {})
    assert result.ok is False
    assert result.error == "unsupported_connector"


def test_connector_test_ig_missing_credentials() -> None:
    result = run_connector_connection_test("ig", "both", {"username": "u"})
    assert result.ok is False
    assert result.error == "missing_credentials"


@patch("chatbot.trader.ig_connector.IgConnector")
def test_connector_test_ig_login_ok(mock_cls) -> None:
    import pandas as pd

    mock_ig = MagicMock()
    mock_ig._cst = "cst"
    mock_ig._security = "sec"
    mock_ig.get_ohlc.return_value = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [8310.5]},
        index=pd.DatetimeIndex(["2026-07-21 12:00:00+02:00"]),
    )
    mock_cls.return_value = mock_ig
    result = run_connector_connection_test(
        "ig",
        "both",
        {
            "api_key": "k",
            "username": "u",
            "password": "p",
            "acc_type": "DEMO",
            "epic": "IX.D.CAC.DAILY.IP",
        },
    )
    assert result.ok is True
    assert "login OK" in result.message
    assert "8310.50" in result.message
    assert "env=DEMO" in result.message
    mock_ig.login.assert_called_once()
    mock_ig.close.assert_called_once()


@patch("chatbot.trader.ig_connector.IgConnector")
def test_connector_test_ig_auth_error_details(mock_cls) -> None:
    from chatbot.trader.ig_connector import IgAuthError

    mock_ig = MagicMock()
    mock_ig.login.side_effect = IgAuthError(
        "IG login failed: HTTP 401\n"
        "URL: https://demo-api.ig.com/gateway/deal/session\n"
        "IG errorCode: error.security.invalid-details\n"
        "Hint: Check username/password (demo login, not email)."
    )
    mock_cls.return_value = mock_ig
    result = run_connector_connection_test(
        "ig",
        "both",
        {
            "api_key": "abcd1234efgh5678",
            "username": "samsam114",
            "password": "secret",
            "acc_type": "DEMO",
            "epic": "IX.D.CAC.DAILY.IP",
        },
    )
    assert result.ok is False
    assert "HTTP 401" in result.message
    assert "error.security.invalid-details" in result.message
    assert "Hint:" in result.message
    assert result.error is not None
    assert "username=samsam114" in result.error
    assert "abcd" in result.error  # masked key prefix
