from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from chatbot.application.monitoring_format import format_count, format_count_compact, format_usd
from chatbot.application.monitoring_series import (
    disk_pie_chart_payload,
    fill_token_series,
    multi_disk_chart_payload,
)
from chatbot.domain.models.api_usage import DiskDayPoint, TokenDayPoint


def test_format_count_large_numbers() -> None:
    assert format_count(1_500_000) == "1,500,000"
    assert format_count_compact(1_500_000) == "1.5M"
    assert format_usd(Decimal("12.345")) == "$12.34"


def test_multi_disk_chart_payload() -> None:
    points_a = [
        DiskDayPoint(snapshot_date=date(2026, 6, 1), total_bytes=100),
        DiskDayPoint(snapshot_date=date(2026, 6, 2), total_bytes=200),
    ]
    points_b = [
        DiskDayPoint(snapshot_date=date(2026, 6, 1), total_bytes=1000),
        DiskDayPoint(snapshot_date=date(2026, 6, 2), total_bytes=1100),
    ]
    payload = multi_disk_chart_payload(("Bots", points_a), ("Host", points_b))
    assert payload["labels"] == ["2026-06-01", "2026-06-02"]
    assert len(payload["series"]) == 2
    assert payload["series"][0]["total_bytes"] == [100, 200]


def test_disk_pie_chart_payload() -> None:
    payload = disk_pie_chart_payload(used_bytes=60, free_bytes=40)
    assert payload == {"used_bytes": 60, "free_bytes": 40}


def test_fill_token_series_zero_fills_gaps() -> None:
    since = date(2026, 6, 1)
    until = date(2026, 6, 3)
    points = [
        TokenDayPoint(usage_date=date(2026, 6, 1), prompt_tokens=10, output_tokens=1),
        TokenDayPoint(usage_date=date(2026, 6, 3), prompt_tokens=5, output_tokens=2),
    ]
    filled = fill_token_series(points, since, until)
    assert len(filled) == 3
    assert filled[1].prompt_tokens == 0
    assert filled[2].output_tokens == 2
