from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from chatbot.application.trader_backtest_service import default_ohlc_path, run_due_ig_ohlc_syncs
from chatbot.trader.config import TraderConfig
from chatbot.trader.ig_connector import IgApiError, IgConnector, _extract_price_allowance
from chatbot.trader.live_ohlc_feed import (
    build_live_frames,
    prepare_live_ohlc_feed,
    top_up_csv_from_connector,
)
from chatbot.trader.ohlc_store import load_ohlc_csv
from chatbot.trader.scheduler import LiveScheduler
from chatbot.config.settings import Settings


def _sample_bars(start: str, n: int = 40) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min", tz="Europe/Paris")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [101.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [100.5 + i * 0.1 for i in range(n)],
            "volume": [10] * n,
        },
        index=idx,
    )


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(path, index=False)


def test_extract_price_allowance() -> None:
    out = _extract_price_allowance(
        {
            "metadata": {
                "allowance": {
                    "remainingAllowance": 1234,
                    "totalAllowance": 10000,
                    "allowanceExpiry": 1710000000,
                }
            }
        }
    )
    assert out is not None
    assert out["remaining"] == 1234
    assert out["total"] == 10000
    assert out["expiry"] == 1710000000
    assert out.get("fetched_at")


def test_present_ig_price_allowance_countdown() -> None:
    from chatbot.trader.ig_allowance import present_ig_price_allowance

    now = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    view = present_ig_price_allowance(
        {
            "remainingAllowance": 100,
            "totalAllowance": 10000,
            "allowanceExpiry": 3600,
            "fetched_at": "2024-06-01T11:00:00+00:00",  # 1h ago → 0 left
        },
        now=now,
    )
    assert view is not None
    assert view["remaining"] == 100
    assert view["resets_in_seconds"] == 0
    assert "resets in now" in view["label"] or view["exhausted"] is False


def test_top_up_appends_newer_bars(tmp_path: Path) -> None:
    path = tmp_path / "ohlc_15m.csv"
    existing = _sample_bars("2024-06-01 09:00:00", n=2)
    _write_csv(path, existing)
    fresh = _sample_bars("2024-06-01 09:00:00", n=4)  # includes 2 newer
    connector = MagicMock(spec=IgConnector)
    connector.get_ohlc.return_value = fresh
    connector.last_price_allowance = {"remainingAllowance": 999, "remaining": 999}

    # Force fetch by making expected closed bar ahead of CSV last.
    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=pd.Timestamp("2024-06-01 10:00:00", tz="Europe/Paris"),
    ):
        result = top_up_csv_from_connector(path, connector, max_bars=8)

    assert result["added"] == 2
    assert result["mode"] == "cheap"
    assert result["allowance"]["remaining"] == 999
    loaded = load_ohlc_csv(path)
    assert len(loaded) == 4
    connector.get_ohlc.assert_called_once_with("15m", 8)


def test_top_up_cheap_discontinuous_falls_back_to_range(tmp_path: Path) -> None:
    """Cheap tip that skips bars must not be appended — range-fill instead."""
    path = tmp_path / "ohlc_15m.csv"
    existing = _sample_bars("2024-06-03 10:00:00", n=1)  # last 10:00
    _write_csv(path, existing)
    # Tip starts at 11:00 — would create a mid-session hole if appended raw.
    tip = _sample_bars("2024-06-03 11:00:00", n=4)
    catchup = _sample_bars("2024-06-03 10:15:00", n=7)  # contiguous from 10:15
    connector = MagicMock(spec=IgConnector)
    connector.get_ohlc.return_value = tip
    connector.fetch_ohlc_range.return_value = catchup
    connector.last_price_allowance = {"remaining": 9000}

    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=pd.Timestamp("2024-06-03 12:00:00", tz="Europe/Paris"),
    ):
        result = top_up_csv_from_connector(
            path,
            connector,
            max_bars=8,
            now=datetime(2024, 6, 3, 10, 10, tzinfo=UTC),
        )

    assert result["mode"] == "range"
    assert result["added"] == 7
    connector.fetch_ohlc_range.assert_called_once()
    loaded = load_ohlc_csv(path)
    assert list(loaded.index) == list(
        pd.date_range("2024-06-03 10:00:00", periods=8, freq="15min", tz="Europe/Paris")
    )


def test_find_intrasession_gaps_detects_mid_session_hole() -> None:
    from chatbot.trader.ohlc_store import find_intrasession_gaps, summarize_ohlc_gaps

    a = _sample_bars("2024-06-03 10:00:00", n=2)  # 10:00, 10:15
    b = _sample_bars("2024-06-03 11:00:00", n=2)  # hole 10:30, 10:45
    df = pd.concat([a, b])
    gaps = find_intrasession_gaps(df)
    assert len(gaps) == 1
    assert gaps[0][0] == pd.Timestamp("2024-06-03 10:15:00", tz="Europe/Paris")
    assert gaps[0][1] == pd.Timestamp("2024-06-03 11:00:00", tz="Europe/Paris")
    report = summarize_ohlc_gaps(df)
    assert report["has_recent_gaps"] is True
    assert report["has_gaps"] is True
    assert report["gap_count"] == 1
    assert report["gaps"][0]["missing_bars_approx"] == 2
    assert report["fix_steps"]
    assert summarize_ohlc_gaps(_sample_bars("2024-06-03 10:00:00", n=8))["has_gaps"] is False

    # Ancient hole only → info, not a live blocker.
    old_a = _sample_bars("2000-06-01 10:00:00", n=2)
    old_b = _sample_bars("2000-06-01 11:00:00", n=2)
    tip = _sample_bars("2024-06-03 10:00:00", n=40)
    hist_only = pd.concat([old_a, old_b, tip])
    hist_report = summarize_ohlc_gaps(hist_only)
    assert hist_report["has_recent_gaps"] is False
    assert hist_report["historical_gap_count"] == 1
    assert hist_report["severity"] == "info"


def test_prepare_feed_warns_but_allows_llm_on_chart_gap(tmp_path: Path) -> None:
    path = tmp_path / "ohlc_15m.csv"
    a = _sample_bars("2024-06-03 09:00:00", n=10)
    b = _sample_bars("2024-06-03 12:00:00", n=40)  # mid-session hole
    _write_csv(path, pd.concat([a, b]))
    # CSV already "fresh" vs expected → no top-up, but chart window gapped.
    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=pd.Timestamp(b.index[-1]),
    ):
        feed = prepare_live_ohlc_feed(
            path,
            config=TraderConfig(
                lookback_15m=80,
                warmup_bars=14,
                chart_show_pivots=False,
            ),
            connector=None,
            top_up=False,
        )
    assert feed.skip_llm is False
    assert feed.error is None
    assert any("mid-session gap" in w for w in feed.warnings)


def test_top_up_range_catchup_fills_large_gap(tmp_path: Path) -> None:
    path = tmp_path / "ohlc_15m.csv"
    existing = _sample_bars("2024-06-01 09:00:00", n=2)  # last 09:15
    _write_csv(path, existing)
    # Simulate a week of missing bars returning from range fetch.
    catchup = _sample_bars("2024-06-01 09:30:00", n=20)
    connector = MagicMock(spec=IgConnector)
    connector.fetch_ohlc_range.return_value = catchup
    connector.last_price_allowance = {"remaining": 8000, "total": 10000}

    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=pd.Timestamp("2024-06-03 12:00:00", tz="Europe/Paris"),
    ):
        result = top_up_csv_from_connector(
            path,
            connector,
            max_bars=8,
            now=datetime(2024, 6, 3, 10, 10, tzinfo=UTC),
        )

    assert result["mode"] == "range"
    assert result["added"] == 20
    connector.get_ohlc.assert_not_called()
    connector.fetch_ohlc_range.assert_called_once()
    loaded = load_ohlc_csv(path)
    assert len(loaded) == 22


def test_top_up_skips_fetch_when_csv_fresh(tmp_path: Path) -> None:
    path = tmp_path / "ohlc_15m.csv"
    # Last bar = expected closed → no IG call.
    expected = pd.Timestamp("2024-06-01 12:00:00", tz="Europe/Paris")
    existing = _sample_bars("2024-06-01 12:00:00", n=1)
    _write_csv(path, existing)
    connector = MagicMock(spec=IgConnector)
    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=expected,
    ):
        result = top_up_csv_from_connector(path, connector)
    assert result["added"] == 0
    assert result["skipped_fetch"] is True
    connector.get_ohlc.assert_not_called()


def test_build_live_frames_resamples_without_ig() -> None:
    df = _sample_bars("2024-06-03 08:00:00", n=96)  # 24h of 15m
    cfg = TraderConfig(
        lookback_15m=20,
        lookback_1h=10,
        lookback_1d=5,
        warmup_bars=14,
        chart_show_pivots=False,
    )
    w15, w1h, w1d = build_live_frames(df, cfg)
    assert not w15.empty
    assert not w1h.empty
    assert not w1d.empty
    assert len(w15) == 20 + 14
    assert len(w1h) <= 10 + 14


def test_prepare_feed_missing_csv(tmp_path: Path) -> None:
    feed = prepare_live_ohlc_feed(tmp_path / "missing.csv", config=TraderConfig())
    assert feed.skip_llm is True
    assert feed.error and "missing" in feed.error.lower()


def test_prepare_feed_stale_ok_on_top_up_fail(tmp_path: Path) -> None:
    path = tmp_path / "ohlc_15m.csv"
    # now 12:10 Paris; last bar 11:45 Paris → ~1.7 slots old
    now = datetime(2024, 6, 1, 10, 10, tzinfo=UTC)
    bars = _sample_bars("2024-06-01 08:00:00", n=16)  # ends 11:45 Paris
    _write_csv(path, bars)
    connector = MagicMock(spec=IgConnector)
    err = IgApiError("historical-data-allowance")
    connector.get_ohlc.side_effect = err
    connector.fetch_ohlc_range.side_effect = err

    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=pd.Timestamp("2024-06-01 12:00:00", tz="Europe/Paris"),
    ):
        feed = prepare_live_ohlc_feed(
            path, config=TraderConfig(), connector=connector, now=now
        )
    assert feed.top_up_ok is False
    assert feed.stale is True
    assert feed.skip_llm is False
    assert not feed.ohlc_15.empty
    assert any("stale_data" in w for w in feed.warnings)


def test_prepare_feed_too_old_skips_llm(tmp_path: Path) -> None:
    path = tmp_path / "ohlc_15m.csv"
    now = datetime(2024, 6, 1, 18, 0, tzinfo=UTC)
    bars = _sample_bars("2024-06-01 09:00:00", n=10)
    _write_csv(path, bars)
    connector = MagicMock(spec=IgConnector)
    err = IgApiError("historical-data-allowance")
    connector.get_ohlc.side_effect = err
    connector.fetch_ohlc_range.side_effect = err

    with patch(
        "chatbot.trader.live_ohlc_feed.expected_last_closed_15m",
        return_value=pd.Timestamp("2024-06-01 19:45:00", tz="Europe/Paris"),
    ):
        feed = prepare_live_ohlc_feed(
            path, config=TraderConfig(), connector=connector, now=now
        )
    assert feed.skip_llm is True
    assert feed.error


def test_scheduler_uses_provider_not_full_ig_lookbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = TraderConfig(
        lookback_15m=20,
        lookback_1h=10,
        lookback_1d=5,
        warmup_bars=14,
        chart_show_pivots=False,
        llm_trigger_mode="interval",
        llm_every_bars=999,
    )
    df = _sample_bars("2024-06-03 08:00:00", n=96)
    w15, w1h, w1d = build_live_frames(df, cfg)

    from chatbot.trader.live_ohlc_feed import LiveOhlcFeed

    feed = LiveOhlcFeed(
        ohlc_15=w15,
        ohlc_1h=w1h,
        ohlc_1d=w1d,
        last_price=float(df["close"].iloc[-1]),
        last_bar_ts=str(df.index[-1]),
        allowance={"remaining": 5000, "remainingAllowance": 5000},
    )
    sched = LiveScheduler(
        cfg,
        api_key="x",
        journal_dir=tmp_path / "journal",
        dry_run=True,
        ohlc_provider=lambda: feed,
    )
    sched.ig._cst = "cst"  # noqa: SLF001 — skip login
    sched.ig.get_ohlc = MagicMock(side_effect=AssertionError("should not call get_ohlc"))
    sched.ig.sync_price = MagicMock(side_effect=AssertionError("should not sync_price"))
    sched.fm.notify = MagicMock()
    sched.llm.decide = MagicMock(return_value=None)
    # Force quiet trigger so we only exercise the OHLC path.
    quiet = MagicMock()
    quiet.should_call = False
    quiet.reasons = ["quiet"]
    sched.trigger.evaluate = MagicMock(return_value=quiet)
    monkeypatch.setattr(
        "chatbot.trader.scheduler.session_snapshot",
        lambda *a, **k: {
            "dealing_open": True,
            "flatten_enabled": False,
            "flatten_now": False,
            "flatten_reason": "",
            "flatten_reasons": [],
            "flatten_close_at": None,
            "flatten_window_start": None,
            "minutes_to_close": None,
            "next_open_day": None,
            "next_open": None,
            "next_close": None,
            "close_id": None,
            "now": "2024-06-03T12:00:00+01:00",
            "weekday": "Monday",
            "tz": "Europe/London",
            "source": "test",
            "weekly_open": "Sun 23:02 Europe/London",
            "weekly_close": "Fri 22:00 Europe/London",
        },
    )

    payload = sched.run_once()
    assert payload["ohlc_feed"]["source"] == "local_csv"
    assert payload["ohlc_feed"]["allowance"]["remaining"] == 5000
    assert float(sched.ig.ledger.last_price) == pytest.approx(feed.last_price)
    assert payload["skipped"] is True
    sched.ig.get_ohlc.assert_not_called()
    sched.llm.decide.assert_not_called()


@patch("chatbot.application.trader_backtest_service.sync_ohlc_from_ig")
def test_run_due_skips_armed_bots(mock_sync, tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    session = MagicMock()
    tenant = MagicMock()
    tenant.id = 1
    tenant.slug = "armed-bot"

    # Seed CSV so missing-csv isn't the skip reason.
    path = default_ohlc_path(settings, "armed-bot")
    _write_csv(path, _sample_bars("2024-06-01 09:00:00", n=3))

    with (
        patch(
            "chatbot.adapters.persistence.tenant_repository.SqlAlchemyTenantRepository"
        ) as mock_tenant_repo,
        patch("chatbot.application.tenant_service.TenantService"),
        patch(
            "chatbot.adapters.persistence.connector_repository.SqlAlchemyConnectorRepository"
        ),
        patch("chatbot.application.connector_service.ConnectorService"),
        patch(
            "chatbot.application.trader_live_service.load_live_config",
            return_value={"mode": "live", "ig_connector_ids": [1]},
        ),
        patch(
            "chatbot.application.trader_live_service.resolve_primary_ig_config",
            return_value={"api_key": "k", "username": "u", "password": "p"},
        ),
    ):
        mock_tenant_repo.return_value.list_active_traders.return_value = [tenant]
        logs = run_due_ig_ohlc_syncs(session, settings)

    assert any("armed (live)" in line for line in logs)
    mock_sync.assert_not_called()
