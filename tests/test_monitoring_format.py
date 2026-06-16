from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from chatbot.application.monitoring_format import format_count, format_count_compact, format_usd
from chatbot.application.monitoring_series import fill_token_series
from chatbot.domain.models.api_usage import TokenDayPoint


def test_format_count_large_numbers() -> None:
    assert format_count(1_500_000) == "1,500,000"
    assert format_count_compact(1_500_000) == "1.5M"
    assert format_usd(Decimal("12.345")) == "$12.34"


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
