from __future__ import annotations

import email
from datetime import UTC, datetime
from email.message import EmailMessage as StdEmailMessage
from email.utils import format_datetime
from unittest.mock import MagicMock, patch

import pytest

from chatbot.adapters.mail.imap_client import (
    IncomingMail,
    ImapMailClient,
    _body_text_from_message,
    _decode_header_value,
    _extract_email_address,
    _imap_use_ssl,
    imap_client,
)
from chatbot.adapters.mail.smtp_sender import SmtpEmailSender, _parse_use_tls
from chatbot.adapters.mail.types import EmailMessage


def test_parse_use_tls_defaults() -> None:
    assert _parse_use_tls(None) is True
    assert _parse_use_tls("") is True
    assert _parse_use_tls(False) is False
    assert _parse_use_tls("false") is False
    assert _parse_use_tls("on") is True


@patch("chatbot.adapters.mail.smtp_sender.smtplib.SMTP")
def test_smtp_sender_no_tls(mock_smtp_cls) -> None:
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__.return_value = mock_smtp
    SmtpEmailSender(
        host="greenmail",
        port=3025,
        username="",
        password="",
        use_tls=False,
    ).send(
        EmailMessage(
            to_addr="bot@test.local",
            subject="Hi",
            body_text="Plain body",
            body_html="<p>HTML body</p>",
            from_addr="client@example.com",
        )
    )
    mock_smtp.starttls.assert_not_called()
    mock_smtp.send_message.assert_called_once()
    sent = mock_smtp.send_message.call_args[0][0]
    assert sent.get_content_type() == "multipart/alternative"
    parts = list(sent.walk())
    payloads = {p.get_content_type(): p.get_content() for p in parts if not p.is_multipart()}
    assert payloads["text/plain"].strip() == "Plain body"
    assert payloads["text/html"].strip() == "<p>HTML body</p>"


@patch("chatbot.adapters.mail.smtp_sender.smtplib.SMTP")
def test_smtp_sender_with_tls(mock_smtp_cls) -> None:
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__.return_value = mock_smtp
    SmtpEmailSender(
        host="smtp.example.com",
        port=587,
        username="user",
        password="pass",
        use_tls=True,
    ).send(
        EmailMessage(
            to_addr="a@b.com",
            subject="Hi",
            body_text="Body",
            from_addr="bot@example.com",
        )
    )
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("user", "pass")


def test_imap_use_ssl_by_port() -> None:
    assert _imap_use_ssl({"imap_port": "3143"}) is False
    assert _imap_use_ssl({"imap_port": "993"}) is True
    assert _imap_use_ssl({"imap_port": "3993"}) is True
    assert _imap_use_ssl({"imap_use_ssl": "true", "imap_port": "3143"}) is True


def test_decode_header_and_address() -> None:
    assert _extract_email_address("Alice <alice@example.com>") == "alice@example.com"
    msg = StdEmailMessage()
    msg.set_content("Hello world")
    assert _body_text_from_message(email.message_from_string(msg.as_string())) == "Hello world"


@patch.object(ImapMailClient, "connect")
@patch.object(ImapMailClient, "close")
def test_imap_client_fetch_unseen(mock_close, mock_connect) -> None:
    client = ImapMailClient(
        {
            "imap_host": "greenmail",
            "imap_port": "3143",
            "username": "bot@test.local",
            "password": "secret",
        }
    )
    client._conn = MagicMock()
    client._conn.uid.side_effect = [
        ("OK", [b"1 2"]),
        ("OK", [(b"1 (BODY[] {..}", _sample_raw_email())]),
        ("OK", [(b"2 (BODY[] {..}", _sample_raw_email(from_addr="bob@test.com"))]),
    ]

    mails = client.fetch_pending(lambda uid: False)
    assert len(mails) == 2
    assert mails[0].from_addr == "client@example.com"
    assert mails[0].body_text == "Need a quote"


def _sample_raw_email(
    *,
    from_addr: str = "client@example.com",
    received_at: datetime | None = None,
) -> bytes:
    msg = StdEmailMessage()
    msg["From"] = from_addr
    msg["To"] = "bot@test.local"
    msg["Subject"] = "Quote"
    if received_at is not None:
        msg["Date"] = format_datetime(received_at)
    msg.set_content("Need a quote")
    return msg.as_bytes()


@patch.object(ImapMailClient, "connect")
@patch.object(ImapMailClient, "close")
def test_imap_client_parses_received_at(mock_close, mock_connect) -> None:
    client = ImapMailClient({"imap_host": "h", "imap_port": "3143", "username": "u", "password": "p"})
    client._conn = MagicMock()
    received = datetime(2026, 6, 8, 10, 0, tzinfo=UTC)
    client._conn.uid.side_effect = [
        ("OK", [b"1"]),
        ("OK", [(b"1 (BODY[] {..}", _sample_raw_email(received_at=received))]),
    ]
    mails = client.fetch_pending(lambda uid: False)
    assert len(mails) == 1
    assert mails[0].received_at == received


@patch.object(ImapMailClient, "connect")
@patch.object(ImapMailClient, "close")
def test_imap_client_fetch_pending_since_date(mock_close, mock_connect) -> None:
    client = ImapMailClient({"imap_host": "h", "imap_port": "3143", "username": "u", "password": "p"})
    client._conn = MagicMock()
    client._conn.uid.return_value = ("OK", [b""])
    client.fetch_pending(lambda uid: False, since_date="08-Jun-2026")
    client._conn.uid.assert_called_once_with("search", None, 'SINCE "08-Jun-2026"')


@patch.object(ImapMailClient, "connect")
@patch.object(ImapMailClient, "close")
def test_imap_client_mark_seen_uses_parenthesized_flags(mock_close, mock_connect) -> None:
    client = ImapMailClient({"imap_host": "h", "imap_port": "3143", "username": "u", "password": "p"})
    client._conn = MagicMock()
    client._conn.uid.return_value = ("OK", [b"1 (FLAGS (\\Seen))"])

    client.mark_seen("1")

    client._conn.uid.assert_called_once_with("store", "1", "+FLAGS", "(\\Seen)")


@patch("chatbot.adapters.mail.imap_client.imaplib.IMAP4_SSL")
def test_imap_client_xoauth2_connect(mock_imap_ssl) -> None:
    mock_conn = MagicMock()
    mock_conn.authenticate.return_value = ("OK", [b"Success"])
    mock_conn.select.return_value = ("OK", [b"1"])
    mock_imap_ssl.return_value = mock_conn
    client = ImapMailClient(
        {
            "imap_host": "imap.gmail.com",
            "imap_port": "993",
            "username": "user@example.com",
            "_resolved_access_token": "tok",
        }
    )
    client.connect()
    mock_conn.authenticate.assert_called_once()
    mock_conn.login.assert_not_called()


@patch("chatbot.adapters.mail.imap_client.ImapMailClient")
def test_imap_client_context_manager(mock_cls) -> None:
    instance = MagicMock()
    mock_cls.return_value = instance
    with imap_client({"imap_host": "h", "username": "u", "password": "p"}) as client:
        assert client is instance
    instance.connect.assert_called_once()
    instance.close.assert_called_once()
