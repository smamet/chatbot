from __future__ import annotations

import email
import email.utils
import imaplib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.header import decode_header
from typing import Callable, Iterator

from chatbot.adapters.mail.xoauth2 import build_xoauth2_string
from chatbot.application.email_body_sanitize import normalize_inbound_body_text


class ImapError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IncomingMail:
    uid: str
    from_addr: str
    to_addr: str
    subject: str
    body_text: str
    received_at: datetime | None = None
    message_id: str = ""
    in_reply_to: str = ""
    references: tuple[str, ...] = ()
    body_html: str | None = None


@dataclass(frozen=True, slots=True)
class InboxPreviewMessage:
    uid: str
    from_addr: str
    to_addr: str
    subject: str
    body_preview: str
    received_at: datetime | None = None


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    parts: list[str] = []
    for chunk, charset in decode_header(value):
        if isinstance(chunk, bytes):
            parts.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(str(chunk))
    return "".join(parts).strip()


def _extract_email_address(header_value: str) -> str:
    parsed = email.utils.parseaddr(header_value)
    return (parsed[1] or header_value).strip().lower()


def _extract_body_html(msg: email.message.Message) -> str | None:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace").strip()
        return None
    if msg.get_content_type() == "text/html":
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace").strip()
    return None


def _parse_message_ids(header_value: str | None) -> tuple[str, ...]:
    if not header_value:
        return ()
    return tuple(
        f"<{chunk.strip('<>')}>"
        for chunk in re.findall(r"<[^>]+>", header_value)
    )


def _normalize_single_message_id(header_value: str | None) -> str:
    ids = _parse_message_ids(header_value)
    return ids[0] if ids else ""


def _body_text_from_message(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return normalize_inbound_body_text(
                        payload.decode(charset, errors="replace")
                    )
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return normalize_inbound_body_text(
                        payload.decode(charset, errors="replace")
                    )
        return ""
    payload = msg.get_payload(decode=True)
    if not payload:
        return ""
    charset = msg.get_content_charset() or "utf-8"
    return normalize_inbound_body_text(payload.decode(charset, errors="replace"))


def _imap_use_ssl(config: dict) -> bool:
    explicit = config.get("imap_use_ssl")
    if explicit is not None and str(explicit).strip() != "":
        return str(explicit).strip().lower() in ("1", "true", "on", "yes")
    port_raw = str(config.get("imap_port", "993")).strip() or "993"
    try:
        port = int(port_raw)
    except ValueError:
        return True
    return port in (993, 3993)


_FETCH_PEEK = "(BODY.PEEK[])"
_SEEN_FLAGS = "(\\Seen)"


class ImapMailClient:
    def __init__(self, config: dict, *, timeout: int = 30) -> None:
        self._config = config
        self._host = str(config.get("imap_host", "")).strip()
        self._port = int(str(config.get("imap_port", "993")).strip() or "993")
        self._username = str(config.get("username", "")).strip()
        self._password = str(config.get("password", "")).strip()
        self._use_ssl = _imap_use_ssl(config)
        self._timeout = timeout
        self._conn: imaplib.IMAP4 | imaplib.IMAP4_SSL | None = None

    def connect(self) -> None:
        if not self._host or not self._username:
            raise ImapError("Missing imap_host or username")
        try:
            if self._use_ssl:
                self._conn = imaplib.IMAP4_SSL(self._host, self._port, timeout=self._timeout)
            else:
                self._conn = imaplib.IMAP4(self._host, self._port, timeout=self._timeout)
            access_token = str(self._config.get("_resolved_access_token", "")).strip()
            if access_token:
                auth_bytes = build_xoauth2_string(self._username, access_token).encode("utf-8")
                typ, _ = self._conn.authenticate("XOAUTH2", lambda _: auth_bytes)
            else:
                typ, _ = self._conn.login(self._username, self._password)
            if typ != "OK":
                raise ImapError("IMAP login failed")
            typ, _ = self._conn.select("INBOX")
            if typ != "OK":
                raise ImapError("IMAP select INBOX failed")
        except imaplib.IMAP4.error as exc:
            raise ImapError(f"IMAP error: {exc}") from exc

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except imaplib.IMAP4.error:
            pass
        try:
            self._conn.logout()
        except imaplib.IMAP4.error:
            pass
        self._conn = None

    def _parse_fetched_mail(self, uid: str, msg_data) -> IncomingMail | None:
        if not msg_data or not msg_data[0]:
            return None
        raw = msg_data[0][1]
        if not isinstance(raw, (bytes, bytearray)):
            return None
        msg = email.message_from_bytes(raw)
        from_addr = _extract_email_address(_decode_header_value(msg.get("From")))
        to_addr = _extract_email_address(_decode_header_value(msg.get("To")))
        subject = _decode_header_value(msg.get("Subject"))
        body = _body_text_from_message(msg)
        body_html = _extract_body_html(msg)
        message_id = _normalize_single_message_id(msg.get("Message-ID"))
        in_reply_to = _normalize_single_message_id(msg.get("In-Reply-To"))
        references = _parse_message_ids(msg.get("References"))
        received_at: datetime | None = None
        date_header = msg.get("Date")
        if date_header:
            try:
                received_at = email.utils.parsedate_to_datetime(date_header)
                if received_at.tzinfo is None:
                    received_at = received_at.replace(tzinfo=UTC)
                else:
                    received_at = received_at.astimezone(UTC)
            except (TypeError, ValueError, OverflowError):
                received_at = None
        if not from_addr or not body:
            return None
        return IncomingMail(
            uid=uid,
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body_text=body,
            received_at=received_at,
            message_id=message_id,
            in_reply_to=in_reply_to,
            references=references,
            body_html=body_html,
        )

    def _parse_fetched_preview(self, uid: str, msg_data) -> InboxPreviewMessage | None:
        if not msg_data or not msg_data[0]:
            return None
        raw = msg_data[0][1]
        if not isinstance(raw, (bytes, bytearray)):
            return None
        msg = email.message_from_bytes(raw)
        from_addr = _extract_email_address(_decode_header_value(msg.get("From")))
        if not from_addr:
            return None
        to_addr = _extract_email_address(_decode_header_value(msg.get("To")))
        subject = _decode_header_value(msg.get("Subject")) or "(no subject)"
        body = _body_text_from_message(msg)
        preview = body[:200] if body else "(no text/plain or text/html body)"
        received_at: datetime | None = None
        date_header = msg.get("Date")
        if date_header:
            try:
                received_at = email.utils.parsedate_to_datetime(date_header)
                if received_at.tzinfo is None:
                    received_at = received_at.replace(tzinfo=UTC)
                else:
                    received_at = received_at.astimezone(UTC)
            except (TypeError, ValueError, OverflowError):
                received_at = None
        return InboxPreviewMessage(
            uid=uid,
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body_preview=preview,
            received_at=received_at,
        )

    def _fetch_uid(self, uid: str) -> IncomingMail | None:
        if self._conn is None:
            raise ImapError("Not connected")
        typ, msg_data = self._conn.uid("fetch", uid, _FETCH_PEEK)
        if typ != "OK":
            return None
        return self._parse_fetched_mail(uid, msg_data)

    def fetch_pending(
        self,
        skip_uid: Callable[[str], bool],
        *,
        since_date: str | None = None,
    ) -> list[IncomingMail]:
        """Fetch inbox messages not yet recorded (by IMAP UID), without marking read."""
        if self._conn is None:
            raise ImapError("Not connected")
        criteria = f'SINCE "{since_date}"' if since_date else "ALL"
        typ, data = self._conn.uid("search", None, criteria)
        if typ != "OK" or not data or not data[0]:
            return []
        mails: list[IncomingMail] = []
        for uid_b in data[0].split():
            uid = uid_b.decode("ascii", errors="replace")
            if skip_uid(uid):
                continue
            mail = self._fetch_uid(uid)
            if mail is not None:
                mails.append(mail)
        return mails

    def fetch_unseen(self) -> list[IncomingMail]:
        if self._conn is None:
            raise ImapError("Not connected")
        typ, data = self._conn.uid("search", None, "UNSEEN")
        if typ != "OK" or not data or not data[0]:
            return []
        mails: list[IncomingMail] = []
        for uid_b in data[0].split():
            uid = uid_b.decode("ascii", errors="replace")
            mail = self._fetch_uid(uid)
            if mail is not None:
                mails.append(mail)
        return mails

    def mark_seen(self, uid: str) -> None:
        if self._conn is None:
            raise ImapError("Not connected")
        typ, _ = self._conn.uid("store", uid, "+FLAGS", _SEEN_FLAGS)
        if typ != "OK":
            raise ImapError(f"Failed to mark UID {uid} as seen")

    def list_recent_messages(self, *, limit: int = 5) -> list[InboxPreviewMessage]:
        """Return the newest messages in INBOX (up to limit), without marking read."""
        if self._conn is None:
            raise ImapError("Not connected")
        if limit < 1:
            return []
        typ, data = self._conn.uid("search", None, "ALL")
        if typ != "OK" or not data or not data[0]:
            return []
        uid_values = [int(uid_b) for uid_b in data[0].split()]
        if not uid_values:
            return []
        uid_values.sort()
        mails: list[InboxPreviewMessage] = []
        for uid_int in reversed(uid_values):
            uid = str(uid_int)
            typ, msg_data = self._conn.uid("fetch", uid, _FETCH_PEEK)
            if typ != "OK":
                continue
            mail = self._parse_fetched_preview(uid, msg_data)
            if mail is not None:
                mails.append(mail)
            if len(mails) >= limit:
                break
        return mails


@contextmanager
def imap_client(config: dict, *, timeout: int = 30) -> Iterator[ImapMailClient]:
    client = ImapMailClient(config, timeout=timeout)
    try:
        client.connect()
        yield client
    finally:
        client.close()
