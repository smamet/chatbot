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
    _extract_email_addresses,
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
            message_id="<test@example.com>",
        )
    )
    mock_smtp.starttls.assert_not_called()
    mock_smtp.sendmail.assert_called_once()
    envelope_from, recipients, raw = mock_smtp.sendmail.call_args[0]
    assert envelope_from == "client@example.com"
    assert recipients == ["bot@test.local"]
    sent = email.message_from_bytes(raw)
    assert sent["From"] == "client@example.com"
    assert sent.get_content_type() == "multipart/alternative"
    parts = list(sent.walk())
    payloads: dict[str, str] = {}
    for part in parts:
        if part.is_multipart():
            continue
        payload = part.get_payload(decode=True)
        if payload is not None:
            payloads[part.get_content_type()] = payload.decode("utf-8", errors="replace")
    assert payloads["text/plain"].strip() == "Plain body"
    assert payloads["text/html"].strip() == "<p>HTML body</p>"


@patch("chatbot.adapters.mail.smtp_sender.smtplib.SMTP")
def test_smtp_sender_xoauth2(mock_smtp_cls) -> None:
    from chatbot.adapters.mail.xoauth2 import build_xoauth2_string

    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__.return_value = mock_smtp
    SmtpEmailSender(
        host="smtp.office365.com",
        port=587,
        username="support@example.com",
        password="",
        use_tls=True,
        access_token="oauth-token",
    ).verify_connection()
    mock_smtp.auth.assert_called_once()
    _mech, auth_cb = mock_smtp.auth.call_args[0]
    assert _mech == "XOAUTH2"
    assert auth_cb() == build_xoauth2_string("support@example.com", "oauth-token")
    mock_smtp.login.assert_not_called()


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
            message_id="<test@example.com>",
        )
    )
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("user", "pass")
    mock_smtp.sendmail.assert_called_once_with(
        "bot@example.com",
        ["a@b.com"],
        mock_smtp.sendmail.call_args[0][2],
    )


@patch("chatbot.adapters.mail.smtp_sender.smtplib.SMTP")
def test_smtp_sender_utf8_accents_use_base64(mock_smtp_cls) -> None:
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__.return_value = mock_smtp
    body_html = (
        "<p>VDtec est spécialisée dans les systèmes de sécurité.</p>"
        "<p>Rue de la Démocratie&nbsp;— test</p>"
    )
    SmtpEmailSender(
        host="smtp.example.com",
        port=587,
        username="user",
        password="pass",
        use_tls=True,
    ).send(
        EmailMessage(
            to_addr="a@b.com",
            subject="Rép. : Question",
            body_text="VDtec est spécialisée dans les systèmes de sécurité.",
            body_html=body_html,
            from_addr="bot@example.com",
            message_id="<test@example.com>",
        )
    )
    raw = mock_smtp.sendmail.call_args[0][2]
    parsed = email.message_from_bytes(raw)
    for part in parsed.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_content_type() not in ("text/plain", "text/html"):
            continue
        assert part.get("Content-Transfer-Encoding") == "base64"
        assert part.get_content_charset() == "utf-8"
        decoded = part.get_payload(decode=True).decode("utf-8")
        assert "spécialisée" in decoded
        if part.get_content_type() == "text/html":
            assert "Démocratie" in decoded
            assert "&nbsp;" in decoded


def test_imap_use_ssl_by_port() -> None:
    assert _imap_use_ssl({"imap_port": "3143"}) is False
    assert _imap_use_ssl({"imap_port": "993"}) is True
    assert _imap_use_ssl({"imap_port": "3993"}) is True
    assert _imap_use_ssl({"imap_use_ssl": "true", "imap_port": "3143"}) is True


def test_decode_header_and_address() -> None:
    assert _extract_email_address("Alice <alice@example.com>") == "alice@example.com"
    assert _extract_email_addresses(
        "Client <client@example.com>, Support <support@example.com>"
    ) == ("client@example.com", "support@example.com")
    msg = StdEmailMessage()
    msg.set_content("Hello world")
    assert _body_text_from_message(email.message_from_string(msg.as_string())) == "Hello world"


def test_imap_client_parses_to_and_cc_addresses() -> None:
    client = ImapMailClient({"imap_host": "h", "imap_port": "3143", "username": "u", "password": "p"})
    msg = StdEmailMessage()
    msg["From"] = "client@example.com"
    msg["To"] = "Primary <primary@example.com>"
    msg["Cc"] = "Bot <bot@test.local>, Observer <observer@example.com>"
    msg["Subject"] = "FYI"
    msg.set_content("Hello")
    mail = client._parse_fetched_mail("1", [(b"1 (BODY[] {..}", msg.as_bytes())])
    assert mail is not None
    assert mail.to_addr == "primary@example.com"
    assert mail.to_addrs == ("primary@example.com",)
    assert mail.cc_addrs == ("bot@test.local", "observer@example.com")


def test_body_text_from_message_plain_part_with_html_source() -> None:
    msg = email.message_from_string(
        "Content-Type: text/plain; charset=utf-8\n\n"
        "<html><head><style>p{color:red}</style></head>"
        "<body><p>Quarantine notice</p></body></html>"
    )
    assert _body_text_from_message(msg) == "Quarantine notice"


def test_body_text_from_message_single_part_html() -> None:
    msg = email.message_from_string(
        "Content-Type: text/html; charset=utf-8\n\n"
        "<html><body><p>Payment reminder</p></body></html>"
    )
    assert _body_text_from_message(msg) == "Payment reminder"


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
    import base64

    from chatbot.adapters.mail.xoauth2 import build_xoauth2_string

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
    _mech, auth_cb = mock_conn.authenticate.call_args[0]
    assert _mech == "XOAUTH2"
    assert auth_cb(b"") == build_xoauth2_string("user@example.com", "tok").encode("utf-8")
    assert base64.b64encode(auth_cb(b"")).decode() != auth_cb(b"").decode()
    mock_conn.login.assert_not_called()


@patch("chatbot.adapters.mail.imap_client.ImapMailClient")
def test_imap_client_context_manager(mock_cls) -> None:
    instance = MagicMock()
    mock_cls.return_value = instance
    with imap_client({"imap_host": "h", "username": "u", "password": "p"}) as client:
        assert client is instance
    instance.connect.assert_called_once()
    instance.close.assert_called_once()


def test_parse_message_ids_from_references() -> None:
    from chatbot.adapters.mail.imap_client import _parse_message_ids

    refs = _parse_message_ids("<a@x.com> <b@x.com>")
    assert refs == ("<a@x.com>", "<b@x.com>")


def test_smtp_build_mime_includes_thread_headers() -> None:
    import email

    from chatbot.adapters.mail.smtp_sender import SmtpEmailSender
    from chatbot.adapters.mail.types import EmailMessage

    sender = SmtpEmailSender(
        host="smtp.example.com",
        port=587,
        username="u",
        password="p",
        use_tls=False,
    )
    raw = sender._build_mime_bytes(
        EmailMessage(
            to_addr="client@example.com",
            subject="Re: Test",
            body_text="Hello",
            from_addr="bot@example.com",
            message_id="<out@example.com>",
            in_reply_to="<in@example.com>",
            references="<in@example.com>",
        )
    )
    msg = email.message_from_bytes(raw)
    assert msg["Message-ID"] == "<out@example.com>"
    assert msg["In-Reply-To"] == "<in@example.com>"
    assert msg["References"] == "<in@example.com>"
