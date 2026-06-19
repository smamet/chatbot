from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chatbot.adapters.mail.factory import build_email_sender
from chatbot.adapters.mail.mailgun_sender import MailgunEmailSender
from chatbot.adapters.mail.mailjet_sender import MailjetEmailSender
from chatbot.adapters.mail.smtp_sender import EmailSendError, SmtpEmailSender
from chatbot.adapters.mail.types import EmailMessage
from chatbot.application.channel_outbound import dispatch_channel_reply
from chatbot.application.email_outbound import send_email_reply
from chatbot.domain.models.connector import ConnectorType
from chatbot.domain.models.connector_schema import (
    EmailOutboundProvider,
    fields_for,
    resolve_email_outbound_provider,
)


def test_resolve_email_outbound_provider_defaults_to_smtp() -> None:
    assert resolve_email_outbound_provider({}) == EmailOutboundProvider.SMTP.value
    assert resolve_email_outbound_provider({"outbound_provider": "mailjet"}) == "mailjet"
    assert resolve_email_outbound_provider({"outbound_provider": "unknown"}) == "smtp"


def test_build_email_sender_smtp() -> None:
    sender = build_email_sender(
        {
            "outbound_provider": "smtp",
            "smtp_host": "smtp.example.com",
            "smtp_port": "587",
            "smtp_username": "user",
            "smtp_password": "pass",
        }
    )
    assert isinstance(sender, SmtpEmailSender)


def test_build_email_sender_mailjet() -> None:
    sender = build_email_sender(
        {
            "outbound_provider": "mailjet",
            "mailjet_api_key": "key",
            "mailjet_api_secret": "secret",
        }
    )
    assert isinstance(sender, MailjetEmailSender)


def test_build_email_sender_mailgun() -> None:
    sender = build_email_sender(
        {
            "outbound_provider": "mailgun",
            "mailgun_api_key": "key",
            "mailgun_domain": "mg.example.com",
            "mailgun_region": "eu",
        }
    )
    assert isinstance(sender, MailgunEmailSender)


def test_build_email_sender_missing_smtp_host() -> None:
    with pytest.raises(EmailSendError, match="smtp_host"):
        build_email_sender({"outbound_provider": "smtp", "smtp_port": "587"})


def test_email_out_fields_filtered_by_provider() -> None:
    smtp_keys = {
        f.key
        for f in fields_for(
            ConnectorType.EMAIL.value, "out", outbound_provider=EmailOutboundProvider.SMTP.value
        )
    }
    assert "outbound_provider" in smtp_keys
    assert "smtp_host" in smtp_keys
    assert "smtp_use_tls" in smtp_keys
    assert "mailjet_api_key" not in smtp_keys

    mailjet_keys = {
        f.key
        for f in fields_for(
            ConnectorType.EMAIL.value, "out", outbound_provider=EmailOutboundProvider.MAILJET.value
        )
    }
    assert "mailjet_api_key" in mailjet_keys
    assert "smtp_host" not in mailjet_keys


@patch("chatbot.application.email_outbound.build_email_sender")
def test_send_email_reply_uses_factory(mock_build) -> None:
    mock_sender = MagicMock()
    mock_build.return_value = mock_sender
    send_email_reply(
        config={
            "outbound_provider": "smtp",
            "from_addr": "bot@example.com",
            "default_subject": "Support reply",
        },
        to_addr="client@example.com",
        body="Hello **world**",
    )
    mock_sender.send.assert_called_once()
    msg = mock_sender.send.call_args[0][0]
    assert isinstance(msg, EmailMessage)
    assert msg.to_addr == "client@example.com"
    assert msg.from_addr == "bot@example.com"
    assert msg.subject == "Support reply"
    assert msg.body_text == "Hello world"
    assert msg.body_html is not None
    assert "<strong>world</strong>" in msg.body_html


@patch("chatbot.application.channel_outbound.send_email_reply")
def test_dispatch_channel_reply_email(mock_send) -> None:
    dispatch_channel_reply(
        channel=ConnectorType.EMAIL.value,
        recipient_id="user@test.com",
        text="Draft body",
        config={"from_addr": "bot@example.com"},
        settings=MagicMock(),
    )
    mock_send.assert_called_once_with(
        config={"from_addr": "bot@example.com"},
        to_addr="user@test.com",
        body="Draft body",
        subject=None,
        body_html=None,
        attachments=None,
    )


def test_build_email_sender_smtp_no_tls() -> None:
    sender = build_email_sender(
        {
            "outbound_provider": "smtp",
            "smtp_host": "greenmail",
            "smtp_port": "3025",
            "smtp_use_tls": "false",
        }
    )
    assert isinstance(sender, SmtpEmailSender)
    assert sender._use_tls is False


def test_build_email_sender_smtp_oauth_access_token_fallback() -> None:
    sender = build_email_sender(
        {
            "outbound_provider": "smtp",
            "smtp_host": "smtp.office365.com",
            "smtp_port": "587",
            "smtp_username": "support@example.com",
            "oauth_access_token": "stored-token",
        }
    )
    assert isinstance(sender, SmtpEmailSender)
    assert sender._access_token == "stored-token"


@patch("chatbot.adapters.mail.mailjet_sender.httpx.Client")
def test_mailjet_sender_posts(mock_client_cls) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value = mock_client

    MailjetEmailSender(api_key="k", api_secret="s").send(
        EmailMessage(
            to_addr="a@b.com",
            subject="Hi",
            body_text="Body",
            from_addr="bot@example.com",
        )
    )
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args.kwargs["auth"] == ("k", "s")
