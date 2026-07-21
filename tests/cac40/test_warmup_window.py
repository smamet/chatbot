from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from chatbot.cac40.backtest_engine import BacktestEngine
from chatbot.cac40.chart_renderer import render_ohlc_chart
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.ohlc_store import slice_ohlc_period, window_asof


def _make_ohlc(n: int = 200, freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="Europe/Paris")
    base = 7000.0
    return pd.DataFrame(
        {
            "open": [base + i * 0.1 for i in range(n)],
            "high": [base + i * 0.1 + 1 for i in range(n)],
            "low": [base + i * 0.1 - 1 for i in range(n)],
            "close": [base + i * 0.1 + 0.2 for i in range(n)],
            "volume": [1000] * n,
        },
        index=idx,
    )


def test_chart_window_uses_full_history_before_period() -> None:
    # ~2 weeks of 15m bars so a 1w trade slice still has prior history for charts.
    full = _make_ohlc(1400)
    trade = slice_ohlc_period(full, "1w")
    assert len(trade) < len(full)
    first_ts = trade.index[0]
    lookback = 96
    window = window_asof(full, first_ts, lookback)
    # History before period start is available from full dataset.
    assert len(window) == lookback
    assert window.index[-1] == first_ts
    assert window.index[0] < first_ts
    # Trade-only slice would not have enough lookback at period start.
    trade_only = window_asof(trade, first_ts, lookback)
    assert len(trade_only) < lookback


def test_rsi_seeded_across_display_window(tmp_path: Path) -> None:
    """With extra seed bars, displayed RSI should have no leading NaN/flat-50 pad."""
    from chatbot.cac40.chart_renderer import _rsi

    df = _make_ohlc(40)
    display = 20
    seed = 14
    window = df.iloc[-(display + seed) :]
    out = tmp_path / "chart.png"
    render_ohlc_chart(
        window, title="t", out_path=out, rsi_period=seed, display_bars=display
    )
    rsi = _rsi(window["close"], period=seed).iloc[-display:]
    assert rsi.notna().all()


def test_charts_only_skips_llm(tmp_path: Path) -> None:
    df = _make_ohlc(80)
    csv_path = tmp_path / "ohlc.csv"
    out = df.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(csv_path, index=False)

    run_dir = tmp_path / "run"
    cfg = Cac40Config(
        backtest_period="all",
        warmup_bars=14,
        lookback_15m=20,
        lookback_1h=10,
        lookback_1d=5,
        llm_every_n=1,
        llm_every_unit="15m",
        llm_every_bars=1,
        llm_mode="charts_only",
        allow_market_orders=False,
    )
    engine = BacktestEngine(cfg, ohlc_path=csv_path, run_dir=run_dir, api_key="")
    with patch("chatbot.cac40.backtest_engine.GeminiDecisionClient") as mock_cls:
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm
        report = engine.run()
        mock_llm.decide.assert_not_called()
    assert report["decisions_count"] > 0
    log = (run_dir / "decisions_log.json").read_text(encoding="utf-8")
    assert "charts_only" in log
    assert "llm_fail_closed" not in log
    assert any((run_dir / "charts").rglob("chart_*.png"))


def test_warmup_skips_llm_until_enough_history(tmp_path: Path) -> None:
    """With tiny full history, engine must not call LLM (warmup not satisfied)."""
    df = _make_ohlc(10)
    csv_path = tmp_path / "ohlc.csv"
    out = df.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(csv_path, index=False)

    run_dir = tmp_path / "run"
    cfg = Cac40Config(
        backtest_period="all",
        warmup_bars=14,
        lookback_15m=20,
        lookback_1h=10,
        lookback_1d=5,
        llm_every_n=1,
        llm_every_unit="15m",
        llm_every_bars=1,
        llm_mode="live",
        allow_market_orders=False,
    )
    engine = BacktestEngine(cfg, ohlc_path=csv_path, run_dir=run_dir, api_key="x")
    with patch("chatbot.cac40.backtest_engine.GeminiDecisionClient") as mock_cls:
        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm
        with patch("chatbot.cac40.backtest_engine.render_multi_timeframe", return_value={}):
            engine.run()
        mock_llm.decide.assert_not_called()


def test_warmup_allows_llm_when_history_sufficient(tmp_path: Path) -> None:
    df = _make_ohlc(80)
    csv_path = tmp_path / "ohlc.csv"
    out = df.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(csv_path, index=False)

    run_dir = tmp_path / "run"
    cfg = Cac40Config(
        backtest_period="all",
        warmup_bars=14,
        lookback_15m=20,
        lookback_1h=10,
        lookback_1d=5,
        llm_every_n=1,
        llm_every_unit="15m",
        llm_every_bars=1,
        llm_mode="live",
        allow_market_orders=False,
    )
    engine = BacktestEngine(cfg, ohlc_path=csv_path, run_dir=run_dir, api_key="x")

    from chatbot.cac40.models import LlmAnalysis, LlmDecision

    fake = LlmDecision(
        analysis=LlmAnalysis(support=7000.0, resistance=7100.0, bias="hold"),
        actions=[],
    )
    with patch("chatbot.cac40.backtest_engine.GeminiDecisionClient") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.decide.return_value = fake
        mock_cls.return_value = mock_llm
        with patch(
            "chatbot.cac40.backtest_engine.render_multi_timeframe",
            return_value={"15m": b"\x89PNG"},
        ) as mock_render:
            report = engine.run()
        assert mock_llm.decide.call_count > 0
        assert mock_render.call_count > 0
        # rsi_period + display_bars passed through
        assert mock_render.call_args.kwargs.get("rsi_period") == 14
        assert mock_render.call_args.kwargs.get("display_bars") == {
            "15m": 20,
            "1H": 10,
            "1D": 5,
        }
        assert mock_render.call_args.kwargs.get("show_rsi") is True
        assert mock_render.call_args.kwargs.get("show_pivots") is True
        assert report["decisions_count"] > 0
