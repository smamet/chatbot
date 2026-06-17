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
