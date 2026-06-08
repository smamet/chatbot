from __future__ import annotations

import email
import imaplib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from email.header import decode_header
from typing import Callable, Iterator


class ImapError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IncomingMail:
    uid: str
    from_addr: str
    to_addr: str
    subject: str
    body_text: str


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


def _body_text_from_message(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace").strip()
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                    return re.sub(r"<[^>]+>", " ", text).strip()
        return ""
    payload = msg.get_payload(decode=True)
    if not payload:
        return ""
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace").strip()


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
        if not from_addr or not body:
            return None
        return IncomingMail(
            uid=uid,
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body_text=body,
        )

    def _fetch_uid(self, uid: str) -> IncomingMail | None:
        if self._conn is None:
            raise ImapError("Not connected")
        typ, msg_data = self._conn.uid("fetch", uid, _FETCH_PEEK)
        if typ != "OK":
            return None
        return self._parse_fetched_mail(uid, msg_data)

    def fetch_pending(self, skip_uid: Callable[[str], bool]) -> list[IncomingMail]:
        """Fetch inbox messages not yet recorded (by IMAP UID), without marking read."""
        if self._conn is None:
            raise ImapError("Not connected")
        typ, data = self._conn.uid("search", None, "ALL")
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


@contextmanager
def imap_client(config: dict, *, timeout: int = 30) -> Iterator[ImapMailClient]:
    client = ImapMailClient(config, timeout=timeout)
    try:
        client.connect()
        yield client
    finally:
        client.close()
