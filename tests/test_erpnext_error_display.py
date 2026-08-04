from __future__ import annotations

from evenor.application.erpnext_error_display import format_erpnext_error_message


def test_format_erpnext_error_message_truncates_microseconds() -> None:
    raw = (
        "QTN-07593 (Quotation) has been modified after you have opened it "
        "(2026-06-15 14:17:39.476598, 2026-06-15 14:18:15.466254). "
        "Please refresh to get the latest document."
    )
    assert format_erpnext_error_message(raw) == (
        "QTN-07593 (Quotation) has been modified after you have opened it "
        "(2026-06-15 14:17:39, 2026-06-15 14:18:15). "
        "Please refresh to get the latest document."
    )


def test_format_erpnext_error_message_empty() -> None:
    assert format_erpnext_error_message(None) == ""
    assert format_erpnext_error_message("") == ""
