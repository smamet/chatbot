from __future__ import annotations

from datetime import UTC, datetime

from evenor.mail.process_since import (
    format_for_datetime_local,
    format_process_since_display,
    imap_since_date,
    parse_from_form,
    parse_process_since,
    process_since_now_iso,
)


def test_parse_and_format_roundtrip() -> None:
    iso = "2026-06-08T12:30:00+00:00"
    local = format_for_datetime_local(iso)
    assert parse_from_form(local).startswith("2026-06-08")


def test_format_for_datetime_local_omits_seconds() -> None:
    local = format_for_datetime_local("2026-06-08T12:30:45+00:00")
    assert local.split("T")[1].count(":") == 1


def test_imap_since_date() -> None:
    dt = datetime(2026, 6, 8, 14, 30, tzinfo=UTC)
    assert imap_since_date(dt) == "08-Jun-2026"


def test_process_since_now_iso_is_utc() -> None:
    assert "+00:00" in process_since_now_iso() or process_since_now_iso().endswith("Z")


def test_parse_process_since_missing() -> None:
    assert parse_process_since({}) is None


def test_format_process_since_display_empty() -> None:
    assert format_process_since_display(None) == "—"
