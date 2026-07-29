"""Unit tests for Lightstreamer session cache, stale rules, OHLC stream path."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from chatbot.application.trader_stream_service import (
    STREAM_GAP_REPAIR_RETRY_SECONDS,
    BotStreamRuntime,
    append_closed_stream_bar,
    evaluate_stream_stale,
    local_ohlc_is_caught_up,
    reconcile_open_book_under_lock,
    stream_book_reconcile_is_fresh,
    stream_is_healthy,
)
from chatbot.trader.ig_session_cache import (
    CachedIgSession,
    clear_session_cache,
    get_cached_session,
    login_with_shared_cache,
    session_cache_key,
    store_cached_session,
)
from chatbot.trader.ig_stream_probe import TickBar
from chatbot.trader.live_ohlc_feed import prepare_live_ohlc_feed
from chatbot.trader.config import TraderConfig


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_session_cache()
    yield
    clear_session_cache()


def test_session_cache_roundtrip() -> None:
    key = session_cache_key(
        api_key="k", username="u", account_id="A1", acc_type="DEMO"
    )
    store_cached_session(
        key,
        CachedIgSession(
            cst="c",
            xst="x",
            lightstreamer_endpoint="https://ls.example",
            account_id="A1",
            obtained_at=time.time(),
        ),
    )
    got = get_cached_session(key)
    assert got is not None
    assert got.cst == "c"
    assert got.account_id == "A1"


def test_login_with_shared_cache_reuses_tokens() -> None:
    from chatbot.trader.ig_connector import IgConnector

    cfg = TraderConfig(
        ig_api_key="k",
        ig_username="u",
        ig_password="p",
        ig_account_id="A1",
        ig_acc_type="DEMO",
    )
    ig = IgConnector(cfg, dry_run=True)
    logins = {"n": 0}

    def _fake_login():
        logins["n"] += 1
        ig._cst = "CST1"
        ig._security = "XST1"
        ig.lightstreamer_endpoint = "https://ls.example"
        ig.current_account_id = "A1"

    with patch.object(ig, "login", side_effect=_fake_login):
        s1 = login_with_shared_cache(ig, force=True)
        assert s1.cst == "CST1"
        assert logins["n"] == 1
        ig2 = IgConnector(cfg, dry_run=True)
        with patch.object(ig2, "login", side_effect=_fake_login):
            s2 = login_with_shared_cache(ig2)
            assert s2.cst == "CST1"
            assert logins["n"] == 1  # cache hit
            assert ig2._cst == "CST1"
    ig.close()
    ig2.close()


def test_evaluate_stream_stale_disconnect_and_quiet_market() -> None:
    now = datetime.now(timezone.utc)
    disconnected = {
        "connected": False,
        "disconnected_at": (now - timedelta(seconds=60)).isoformat(),
    }
    stale, reason = evaluate_stream_stale(disconnected, dealing_open=True, now=now)
    assert stale and reason == "disconnected"

    quiet_closed = {
        "connected": True,
        "last_tick_at": (now - timedelta(seconds=600)).isoformat(),
        "market_state": "CLOSED",
        "last_heartbeat_at": now.isoformat(),
    }
    stale2, reason2 = evaluate_stream_stale(quiet_closed, dealing_open=False, now=now)
    assert not stale2 and reason2 is None

    quiet_open = {
        "connected": True,
        "last_tick_at": (now - timedelta(seconds=600)).isoformat(),
        "market_state": "TRADEABLE",
        "last_heartbeat_at": now.isoformat(),
    }
    stale3, reason3 = evaluate_stream_stale(quiet_open, dealing_open=True, now=now)
    assert stale3 and reason3 == "no_ticks"


def test_stream_book_reconcile_is_fresh() -> None:
    now = datetime.now(timezone.utc)
    status = {
        "connected": True,
        "stale": False,
        "last_tick_at": now.isoformat(),
        "last_reconcile_at": now.isoformat(),
        "market_state": "TRADEABLE",
    }
    assert stream_book_reconcile_is_fresh(status, now=now)
    status["last_reconcile_at"] = (now - timedelta(seconds=30)).isoformat()
    assert not stream_book_reconcile_is_fresh(status, now=now)


def test_append_closed_stream_bar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chatbot.config.settings import Settings

    settings = Settings(data_root=tmp_path)
    slug = "bot-a"
    ohlc_dir = tmp_path / "trader" / slug / "ohlc"
    ohlc_dir.mkdir(parents=True)
    path = ohlc_dir / "ohlc_15m.csv"
    # Seed two contiguous bars
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-29 10:00:00", tz="Europe/Paris"),
            pd.Timestamp("2026-07-29 10:15:00", tz="Europe/Paris"),
        ],
        name="ts",
    )
    seed = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [101.0, 101.5],
            "volume": [1, 1],
        },
        index=idx,
    )
    out = seed.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(path, index=False)

    bar = TickBar(
        bucket_start=datetime(2026, 7, 29, 8, 30, tzinfo=timezone.utc),  # 10:30 Paris
        open=101.5,
        high=103.0,
        low=101.0,
        close=102.5,
        ticks=5,
    )
    result = append_closed_stream_bar(
        settings, slug, bar, timezone_name="Europe/Paris"
    )
    assert result.get("added") == 1, result
    loaded = pd.read_csv(path)
    assert len(loaded) == 3


def test_prepare_live_ohlc_skips_prices_when_stream_healthy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ohlc.csv"
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-29 10:00:00", tz="Europe/Paris")],
        name="ts",
    )
    # Use a recent bar relative to "now" so we don't trip behind-expected without top-up.
    now = datetime(2026, 7, 29, 8, 10, tzinfo=timezone.utc)  # 10:10 Paris → expected 10:00
    seed = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1],
        },
        index=idx,
    )
    out = seed.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(path, index=False)

    connector = MagicMock()
    connector.get_ohlc = MagicMock(side_effect=AssertionError("should not call /prices"))
    cfg = TraderConfig(data_timezone="Europe/Paris")
    feed = prepare_live_ohlc_feed(
        path,
        config=cfg,
        connector=connector,
        top_up=True,
        now=now,
        stream_healthy=True,
        stream_stale=False,
        stream_mid=100.7,
    )
    assert feed.last_price == 100.7
    assert feed.skip_llm is False
    connector.get_ohlc.assert_not_called()


def test_prepare_live_ohlc_fail_closed_when_stream_stale(tmp_path: Path) -> None:
    path = tmp_path / "ohlc.csv"
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-29 10:00:00", tz="Europe/Paris")],
        name="ts",
    )
    seed = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1]},
        index=idx,
    )
    out = seed.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(path, index=False)
    connector = MagicMock()
    feed = prepare_live_ohlc_feed(
        path,
        config=TraderConfig(data_timezone="Europe/Paris"),
        connector=connector,
        top_up=True,
        now=datetime(2026, 7, 29, 8, 10, tzinfo=timezone.utc),
        stream_healthy=False,
        stream_stale=True,
        stream_error="no_ticks",
    )
    assert feed.skip_llm is True
    assert feed.stale is True
    assert "no_ticks" in (feed.error or "")


def test_reconcile_calls_replace_open_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chatbot.config.settings import Settings

    settings = Settings(data_root=tmp_path)
    slug = "bot-b"
    (tmp_path / "trader" / slug / "live").mkdir(parents=True)

    sync_calls: list[dict] = []

    def _fake_sync(ledger, *, positions, working_orders, epic, **kw):
        sync_calls.append({"positions": positions, "orders": working_orders, "epic": epic})
        return {
            "changed": True,
            "imported": [],
            "closed": [],
            "opened": [],
            "order_book": {},
            "warnings": [],
        }

    class _FakeIg:
        def __init__(self, *a, **k):
            self._cst = "c"
            self.config = MagicMock(epic="IX.D.CAC.BMU.IP")

        def list_open_positions(self, epic=""):
            return [{"dealId": "P1"}]

        def list_working_orders(self):
            return [{"dealId": "W1"}]

        def close(self):
            return None

    monkeypatch.setattr(
        "chatbot.application.trader_stream_service.sync_open_book_from_ig",
        _fake_sync,
    )
    monkeypatch.setattr(
        "chatbot.application.trader_stream_service.IgConnector",
        _FakeIg,
    )
    monkeypatch.setattr(
        "chatbot.application.trader_stream_service.login_with_shared_cache",
        lambda ig, **kw: None,
    )

    cfg = TraderConfig(epic="IX.D.CAC.BMU.IP")
    result = reconcile_open_book_under_lock(
        settings,
        slug,
        ig_config={"_connector_id": 7},
        cfg=cfg,
        blocking=True,
    )
    assert result["ok"] is True
    assert result["acquired"] is True
    assert len(sync_calls) == 1
    assert sync_calls[0]["orders"][0]["dealId"] == "W1"


def test_stream_is_healthy_helper() -> None:
    now = datetime.now(timezone.utc)
    assert not stream_is_healthy({})
    assert stream_is_healthy(
        {
            "connected": True,
            "stale": False,
            "last_tick_at": now.isoformat(),
            "market_state": "TRADEABLE",
        },
        dealing_open=True,
    )


def _seed_ohlc_csv(path: Path, ts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.DatetimeIndex([pd.Timestamp(ts, tz="Europe/Paris")], name="ts")
    seed = pd.DataFrame(
        {
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
            "volume": [1],
        },
        index=idx,
    )
    out = seed.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(path, index=False)


def _runtime_with_fake_stream(
    tmp_path: Path, *, slug: str = "fx-bot"
) -> BotStreamRuntime:
    from chatbot.config.settings import Settings

    settings = Settings(data_root=tmp_path)
    cfg = TraderConfig(
        epic="CS.D.EURUSD.MINI.IP",
        data_timezone="Europe/Paris",
        ig_api_key="k",
        ig_username="u",
        ig_password="p",
        ig_account_id="A1",
        ig_acc_type="DEMO",
    )
    rt = BotStreamRuntime(
        settings=settings,
        slug=slug,
        mode="live",
        ig_config={"account_id": "A1", "_connector_id": 1},
        cfg=cfg,
        enable_trade_reconcile=False,
    )
    now = datetime.now(timezone.utc)
    svc = MagicMock()
    svc.connected = True
    svc.status = "CONNECTED:WS-STREAMING"
    svc.last_tick_at = now.isoformat()
    svc.last_bar_closed_at = now.isoformat()
    svc.last_trade_at = None
    svc.ticks_total = 10
    svc.bars_closed_total = 1
    svc.trade_events_total = 0
    svc.reconnect_count = 0
    svc.last_quote = MagicMock(market_state="TRADEABLE")
    svc.last_error = None
    rt._service = svc
    return rt


def test_local_ohlc_is_caught_up(tmp_path: Path) -> None:
    from chatbot.config.settings import Settings
    from chatbot.application.trader_backtest_service import default_ohlc_path

    settings = Settings(data_root=tmp_path)
    slug = "fx-bot"
    path = default_ohlc_path(settings, slug)
    now = datetime(2026, 7, 29, 14, 52, tzinfo=timezone.utc)  # 16:52 Paris → expected 16:45
    _seed_ohlc_csv(path, "2026-07-29 16:45:00")
    assert local_ohlc_is_caught_up(
        settings, slug, timezone_name="Europe/Paris", now=now
    )
    _seed_ohlc_csv(path, "2026-07-29 16:00:00")
    assert not local_ohlc_is_caught_up(
        settings, slug, timezone_name="Europe/Paris", now=now
    )


def test_heartbeat_auto_clears_gap_when_csv_already_synced(tmp_path: Path) -> None:
    """Data-tab Sync should unblock LLM without a REST repair or worker restart."""
    from chatbot.application.trader_backtest_service import default_ohlc_path

    rt = _runtime_with_fake_stream(tmp_path)
    now = datetime.now(timezone.utc)
    # Seed CSV at the current closed 15m slot so caught-up check passes.
    local = pd.Timestamp(now).tz_convert("Europe/Paris").floor("15min") - pd.Timedelta(
        minutes=15
    )
    _seed_ohlc_csv(default_ohlc_path(rt.settings, rt.slug), local.strftime("%Y-%m-%d %H:%M:%S"))

    rt._need_gap_repair = True
    rt._gap_repair_last_error = "previous failure"
    rt._status["error"] = "gap_repair:previous failure"

    with patch(
        "chatbot.application.trader_stream_service.repair_ohlc_gap_via_rest"
    ) as repair:
        status = rt.heartbeat(dealing_open=True)
        repair.assert_not_called()

    assert rt._need_gap_repair is False
    assert status["stale"] is False
    assert status["ok"] is True
    assert status.get("gap_repair_result") == "already_caught_up"
    assert "error" not in status or not status.get("error")


def test_heartbeat_retries_failed_gap_repair_after_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed gap repair must not permanently block recovery (sticky-flag bug)."""
    from chatbot.application.trader_backtest_service import default_ohlc_path

    rt = _runtime_with_fake_stream(tmp_path)
    # Behind CSV so caught-up check fails and REST is attempted.
    _seed_ohlc_csv(
        default_ohlc_path(rt.settings, rt.slug), "2020-01-01 10:00:00"
    )
    rt._need_gap_repair = True
    rt._was_tick_stale = False

    calls: list[int] = []

    def _fail_then_ok(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            return {"added": 0, "error": "allowance_exhausted"}
        return {"added": 2}

    monkeypatch.setattr(
        "chatbot.application.trader_stream_service.repair_ohlc_gap_via_rest",
        _fail_then_ok,
    )
    # First attempt fails → still stale, but flag stays for retry.
    status1 = rt.heartbeat(dealing_open=True)
    assert status1["stale"] is True
    assert status1["stale_reason"] == "gap_repair_failed"
    assert rt._need_gap_repair is True
    assert len(calls) == 1

    # Within backoff → no second REST call.
    status2 = rt.heartbeat(dealing_open=True)
    assert rt._need_gap_repair is True
    assert len(calls) == 1
    assert status2["stale"] is True

    # After backoff → retries and clears.
    rt._last_gap_repair_mono = time.monotonic() - (STREAM_GAP_REPAIR_RETRY_SECONDS + 1)
    status3 = rt.heartbeat(dealing_open=True)
    assert len(calls) == 2
    assert rt._need_gap_repair is False
    assert status3["stale"] is False
    assert status3["ok"] is True
    assert str(status3.get("gap_repair_result") or "").startswith("added=")


def test_heartbeat_queues_repair_after_tick_recovery(tmp_path: Path) -> None:
    from chatbot.application.trader_backtest_service import default_ohlc_path

    rt = _runtime_with_fake_stream(tmp_path)
    _seed_ohlc_csv(
        default_ohlc_path(rt.settings, rt.slug), "2020-01-01 10:00:00"
    )
    rt._was_tick_stale = True
    rt._need_gap_repair = False

    with patch(
        "chatbot.application.trader_stream_service.repair_ohlc_gap_via_rest",
        return_value={"added": 1},
    ) as repair:
        # After repair, mark CSV caught up so clear succeeds even if added alone
        # wouldn't satisfy a second caught-up check in edge cases.
        def _repair_and_fill(*_a, **_k):
            local = (
                pd.Timestamp(datetime.now(timezone.utc))
                .tz_convert("Europe/Paris")
                .floor("15min")
                - pd.Timedelta(minutes=15)
            )
            _seed_ohlc_csv(
                default_ohlc_path(rt.settings, rt.slug),
                local.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return {"added": 3}

        repair.side_effect = _repair_and_fill
        status = rt.heartbeat(dealing_open=True)
        repair.assert_called_once()

    assert status["ok"] is True
    assert rt._need_gap_repair is False
    assert rt._was_tick_stale is False
