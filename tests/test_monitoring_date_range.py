from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi import HTTPException

from chatbot.adapters.persistence.disk_usage_repository import today_utc
from chatbot.application.monitoring_date_range import (
    MONITORING_DEFAULT_DAYS,
    default_monitoring_range,
    monitoring_query_string,
    parse_monitoring_range,
)


def test_default_monitoring_range_is_30_days_inclusive() -> None:
    parsed = default_monitoring_range()
    assert parsed.usage_days == MONITORING_DEFAULT_DAYS
    assert parsed.usage_page == 1
    assert parsed.until == today_utc()
    assert parsed.since == parsed.until - timedelta(days=MONITORING_DEFAULT_DAYS - 1)


def test_parse_monitoring_range_defaults_without_params() -> None:
    parsed = parse_monitoring_range({})
    assert parsed.usage_days == 30
    assert parsed.usage_page == 1


def test_parse_monitoring_range_custom_window() -> None:
    parsed = parse_monitoring_range(
        {"usage_from": "2026-01-01", "usage_to": "2026-01-07", "usage_page": "2"}
    )
    assert parsed.since == date(2026, 1, 1)
    assert parsed.until == date(2026, 1, 7)
    assert parsed.usage_days == 7
    assert parsed.usage_page == 2


def test_parse_monitoring_range_rejects_partial_params() -> None:
    with pytest.raises(HTTPException) as exc:
        parse_monitoring_range({"usage_from": "2026-01-01"})
    assert exc.value.status_code == 422


def test_parse_monitoring_range_rejects_inverted_range() -> None:
    with pytest.raises(HTTPException) as exc:
        parse_monitoring_range({"usage_from": "2026-01-10", "usage_to": "2026-01-01"})
    assert exc.value.status_code == 422


def test_monitoring_query_string_builds_expected_params() -> None:
    qs = monitoring_query_string(
        since=date(2026, 1, 1),
        until=date(2026, 1, 7),
        usage_page=2,
        tab="monitoring",
    )
    assert qs == "tab=monitoring&usage_from=2026-01-01&usage_to=2026-01-07&usage_page=2"
