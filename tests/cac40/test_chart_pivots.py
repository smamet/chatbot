from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from chatbot.cac40.chart_renderer import (
    daily_pivot_map,
    normalize_pivot_period,
    pivot_map,
    render_multi_timeframe,
    render_ohlc_chart,
    traditional_pivots,
)


def test_traditional_pivots_tradingview_formula() -> None:
    levels = traditional_pivots(100, 90, 95)
    assert levels["P"] == 95.0
    assert levels["R1"] == 100.0
    assert levels["S1"] == 90.0
    assert levels["R2"] == 105.0
    assert levels["S2"] == 85.0
    assert levels["R3"] == 110.0
    assert levels["S3"] == 80.0


def test_daily_pivot_map_uses_prior_session() -> None:
    idx = pd.date_range("2024-01-02 09:00", periods=16, freq="1h", tz="Europe/Paris")
    highs = [110 if t.day == 2 else 120 for t in idx]
    lows = [100 if t.day == 2 else 108 for t in idx]
    closes = [105 if t.day == 2 else 115 for t in idx]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1] * len(idx),
        },
        index=idx,
    )
    pivots = daily_pivot_map(df)
    assert date(2024, 1, 3) in pivots
    assert date(2024, 1, 2) not in pivots
    assert pivots[date(2024, 1, 3)]["P"] == traditional_pivots(110, 100, 105)["P"]


def test_normalize_pivot_period() -> None:
    assert normalize_pivot_period("daily") == "D"
    assert normalize_pivot_period("W") == "W"
    assert normalize_pivot_period("monthly") == "M"
    assert normalize_pivot_period("nope") == "D"


def test_weekly_pivot_map() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="1D", tz="Europe/Paris")
    df = pd.DataFrame(
        {
            "open": [100 + i for i in range(20)],
            "high": [105 + i for i in range(20)],
            "low": [95 + i for i in range(20)],
            "close": [102 + i for i in range(20)],
        },
        index=idx,
    )
    pivots = pivot_map(df, "W")
    assert len(pivots) >= 1


def test_render_multi_skips_pivots_on_daily(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="15min", tz="Europe/Paris")
    base = 7000.0
    df15 = pd.DataFrame(
        {
            "open": [base] * 40,
            "high": [base + 1] * 40,
            "low": [base - 1] * 40,
            "close": [base] * 40,
        },
        index=idx,
    )
    df1d = df15.resample("1D").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna()

    with patch("chatbot.cac40.chart_renderer.render_ohlc_chart", return_value=b"\x89PNG") as mock_render:
        render_multi_timeframe(
            {"15m": df15, "1D": df1d},
            out_dir=tmp_path,
            show_pivots=True,
            pivot_period="D",
        )
        calls = {c.kwargs.get("title"): c.kwargs.get("show_pivots") for c in mock_render.call_args_list}
        # titles are "CAC40 15m" / "CAC40 1D"
        assert any(k and "15m" in k and v is True for k, v in calls.items())
        assert any(k and "1D" in k and v is False for k, v in calls.items())


def test_render_respects_rsi_and_pivot_toggles(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="15min", tz="Europe/Paris")
    base = 7000.0
    df = pd.DataFrame(
        {
            "open": [base + i * 0.1 for i in range(80)],
            "high": [base + i * 0.1 + 1 for i in range(80)],
            "low": [base + i * 0.1 - 1 for i in range(80)],
            "close": [base + i * 0.1 + 0.2 for i in range(80)],
            "volume": [1000] * 80,
        },
        index=idx,
    )
    both = render_ohlc_chart(df, title="t", out_path=tmp_path / "both.png", show_rsi=True, show_pivots=True)
    no_rsi = render_ohlc_chart(df, title="t", out_path=tmp_path / "no_rsi.png", show_rsi=False, show_pivots=True)
    assert both[:8] == b"\x89PNG\r\n\x1a\n"
    assert no_rsi[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(both) != len(no_rsi)
