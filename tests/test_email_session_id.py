from __future__ import annotations

from evenor.application.customer_access_gate import parse_session_identity, session_display_label
from evenor.application.email_session_id import (
    build_email_thread_session_id,
    parse_email_thread_session_id,
)


def test_build_and_parse_thread_session_id() -> None:
    sid = build_email_thread_session_id("Client@Example.com", "abc123def456")
    assert sid == "email:client@example.com~abc123def456"
    email, key = parse_email_thread_session_id(sid)
    assert email == "client@example.com"
    assert key == "abc123def456"


def test_legacy_session_id_without_thread() -> None:
    email, key = parse_email_thread_session_id("email:client@example.com")
    assert email == "client@example.com"
    assert key is None


def test_parse_session_identity_strips_thread_key() -> None:
    email, phone = parse_session_identity("email:client@example.com~threadkey1")
    assert email == "client@example.com"
    assert phone is None


def test_session_display_label_strips_thread_key() -> None:
    assert session_display_label("email:client@example.com~threadkey1") == "client@example.com"
