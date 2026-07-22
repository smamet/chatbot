"""Weekend / holiday flatten window + auto-hedge protection."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.llm_decision import build_user_payload
from chatbot.cac40.market_calendar import (
    euronext_closures,
    flatten_check,
    is_trading_day,
)
from chatbot.cac40.models import (
    LegRole,
    LlmAction,
    LlmAnalysis,
    LlmDecision,
    MarketSnapshot,
    OrderPurpose,
    OrderType,
    Side,
    WorkingOrder,
)
from chatbot.cac40.risk_gate import RiskGate
from chatbot.cac40.scheduler import LiveScheduler


PARIS = ZoneInfo("Europe/Paris")


def test_euronext_closures_include_easter_and_skip_bastille():
    c2026 = euronext_closures(2026)
    assert date(2026, 1, 1) in c2026
    assert date(2026, 4, 3) in c2026  # Good Friday 2026
    assert date(2026, 4, 6) in c2026  # Easter Monday 2026
    assert date(2026, 5, 1) in c2026
    assert date(2026, 12, 25) in c2026
    assert date(2026, 12, 26) in c2026
    # Bastille Day — Euronext open
    assert date(2026, 7, 14) not in c2026
    assert is_trading_day(date(2026, 7, 14))


def test_flatten_window_friday_lead_30():
    # Friday 2026-07-24 21:29 Paris — inactive
    assert flatten_check(
        datetime(2026, 7, 24, 21, 29, tzinfo=PARIS),
        close_hhmm="22:00",
        lead_minutes=30,
    )["active"] is False
    # 21:30 — active (weekend)
    fri = flatten_check(
        datetime(2026, 7, 24, 21, 30, tzinfo=PARIS),
        close_hhmm="22:00",
        lead_minutes=30,
    )
    assert fri["active"] is True
    assert "weekend" in fri["reason"]
    # After close — inactive
    assert flatten_check(
        datetime(2026, 7, 24, 22, 1, tzinfo=PARIS),
        close_hhmm="22:00",
        lead_minutes=30,
    )["active"] is False


def test_flatten_window_thursday_before_friday_holiday():
    # Labour Day 2026 is Friday May 1 → flatten Thursday Apr 30 evening
    check = flatten_check(
        datetime(2026, 4, 30, 21, 45, tzinfo=PARIS),
        close_hhmm="22:00",
        lead_minutes=30,
    )
    assert check["active"] is True
    assert "holiday: Labour Day" in check["reason"]


def test_flatten_friday_before_easter_monday_reports_weekend_and_holiday():
    # Good Friday 2026-04-03, Easter Monday 2026-04-06.
    # Thursday 2026-04-02 before Good Friday holiday.
    check = flatten_check(
        datetime(2026, 4, 2, 21, 40, tzinfo=PARIS),
        close_hhmm="22:00",
        lead_minutes=30,
    )
    assert check["active"] is True
    assert "holiday: Good Friday" in check["reason"]


def test_payload_includes_market_clock_and_flatten_instruction():
    snap = MarketSnapshot(symbol="CAC40", last_price=7500.0, phase="LongAtSupport")
    clock = {
        "flatten_now": True,
        "reason": "weekend",
        "net_exposure": 2.0,
        "weekday": "Friday",
        "now": "2026-07-24T21:45:00+02:00",
    }
    raw = build_user_payload(snap, "LongAtSupport", market_clock=clock, order_size=1.0)
    assert "market_clock" in raw
    assert "flatten_now" in raw
    assert "FLATTEN WINDOW ACTIVE" in raw
    assert "size=2" in raw


def test_risk_gate_flatten_allows_market_hedge_uncapped_at_max():
    cfg = Cac40Config(
        allow_market_orders=False,
        max_open_positions=2,
        order_size=1.0,
        spread_points=0,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.market_open(Side.BUY, 1.0)
    ledger.market_open(Side.BUY, 1.0)  # at max
    assert ledger.legs_count() == 2
    assert ledger.net_size() == pytest.approx(2.0)

    gate = RiskGate(cfg, ledger, flatten_active=True)
    result = gate.apply(
        LlmDecision(
            analysis=LlmAnalysis(support=99, resistance=101, bias="hold"),
            actions=[
                LlmAction(
                    op="market_open",
                    side="SELL",
                    size=2.0,
                    purpose="hedge_cover",
                    reason="pre_close",
                )
            ],
        )
    )
    assert result.executed
    assert not any("market_disabled" in r for r in result.rejected)
    assert not any("max_positions" in r for r in result.rejected)
    assert ledger.net_size() == pytest.approx(0.0)
    hedge = [p for p in ledger.positions.values() if p.side == Side.SELL][0]
    assert hedge.size == pytest.approx(2.0)
    assert hedge.role == LegRole.HEDGE


def test_risk_gate_still_blocks_market_without_flatten():
    cfg = Cac40Config(allow_market_orders=False, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.market_open(Side.BUY, 1.0)
    gate = RiskGate(cfg, ledger, flatten_active=False)
    result = gate.apply(
        LlmDecision(
            analysis=LlmAnalysis(bias="hold"),
            actions=[
                LlmAction(op="market_open", side="SELL", size=1, purpose="hedge_cover")
            ],
        )
    )
    assert any("market_disabled" in r for r in result.rejected)


def _ohlc_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-07-24 10:00", periods=n, freq="15min", tz=PARIS)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1] * n,
        },
        index=idx,
    )


def test_auto_flatten_on_llm_fail_closes_net_and_cancels_entries(tmp_path: Path, monkeypatch):
    cfg = Cac40Config(
        flatten_before_close=True,
        flatten_lead_minutes=30,
        market_close_paris="22:00",
        allow_market_orders=False,
        order_size=1.0,
        spread_points=0,
        llm_trigger_mode="interval",
        llm_every_n=1,
        llm_every_unit="15m",
        chart_show_pivots=False,
        chart_show_rsi=False,
    )
    sched = LiveScheduler(cfg, api_key="", journal_dir=tmp_path, dry_run=True)
    sched.ig.ledger.last_price = 100
    sched.ig.ledger.market_open(Side.BUY, 1.0)
    sched.ig.ledger.market_open(Side.BUY, 1.0)
    sched.ig.ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=98,
            size=1,
            purpose=OrderPurpose.ENTRY,
        )
    )
    assert sched.ig.ledger.net_size() == pytest.approx(2.0)
    assert sched.ig.ledger.entry_order_ids()

    df = _ohlc_df()
    sched.ohlc_provider = lambda: type(
        "F",
        (),
        {
            "ohlc_15": df,
            "ohlc_1h": df.iloc[:0],
            "ohlc_1d": df.iloc[:0],
            "last_price": 100.5,
            "last_bar_ts": str(df.index[-1]),
            "top_up_added": 0,
            "top_up_ok": True,
            "stale": False,
            "skip_llm": True,  # force skip LLM → auto path
            "warnings": [],
            "error": None,
            "allowance": None,
        },
    )()

    # Force flatten window active regardless of wall clock.
    monkeypatch.setattr(
        "chatbot.cac40.scheduler.flatten_check",
        lambda *a, **k: {
            "active": True,
            "reason": "weekend",
            "reasons": ["weekend"],
            "close_at": "2026-07-24T22:00:00+02:00",
            "window_start": "2026-07-24T21:30:00+02:00",
            "minutes_to_close": 15,
            "next_open_day": "2026-07-27",
            "now": "2026-07-24T21:45:00+02:00",
            "weekday": "Friday",
            "tz": "Europe/Paris",
        },
    )

    payload = sched.run_once()
    assert payload["market_clock"]["flatten_now"] is True
    assert payload["auto_flatten"] is not None
    assert payload["auto_flatten"]["hedged"] is True
    assert payload["auto_flatten"]["size"] == pytest.approx(2.0)
    assert sched.ig.ledger.net_size() == pytest.approx(0.0)
    assert not sched.ig.ledger.entry_order_ids()


def test_auto_flatten_noop_when_already_flat(tmp_path: Path, monkeypatch):
    cfg = Cac40Config(
        flatten_before_close=True,
        flatten_lead_minutes=30,
        market_close_paris="22:00",
        spread_points=0,
        chart_show_pivots=False,
        chart_show_rsi=False,
    )
    sched = LiveScheduler(cfg, api_key="", journal_dir=tmp_path, dry_run=True)
    sched.ig.ledger.last_price = 100
    sched.ig.ledger.market_open(Side.BUY, 1.0, role=LegRole.PRIMARY)
    sched.ig.ledger.market_open(Side.SELL, 1.0, role=LegRole.HEDGE)
    assert sched.ig.ledger.net_size() == pytest.approx(0.0)

    df = _ohlc_df()
    sched.ohlc_provider = lambda: type(
        "F",
        (),
        {
            "ohlc_15": df,
            "ohlc_1h": df.iloc[:0],
            "ohlc_1d": df.iloc[:0],
            "last_price": 100.5,
            "last_bar_ts": str(df.index[-1]),
            "top_up_added": 0,
            "top_up_ok": True,
            "stale": False,
            "skip_llm": True,
            "warnings": [],
            "error": None,
            "allowance": None,
        },
    )()
    monkeypatch.setattr(
        "chatbot.cac40.scheduler.flatten_check",
        lambda *a, **k: {
            "active": True,
            "reason": "weekend",
            "reasons": ["weekend"],
            "close_at": "x",
            "window_start": "y",
            "minutes_to_close": 10,
            "next_open_day": "z",
            "now": "n",
            "weekday": "Friday",
            "tz": "Europe/Paris",
        },
    )
    payload = sched.run_once()
    assert payload["auto_flatten"] is None or payload["auto_flatten"].get("hedged") is False
    assert sched.ig.ledger.net_size() == pytest.approx(0.0)
    assert len(sched.ig.ledger.positions) == 2


def test_open_market_position_dry_run_sets_hedge_role():
    from chatbot.cac40.ig_connector import IgConnector

    cfg = Cac40Config(spread_points=0)
    conn = IgConnector(cfg, dry_run=True)
    conn.ledger.last_price = 100
    conn.ledger.market_open(Side.BUY, 1.0)
    pid = conn.open_market_position(Side.SELL, 3.0, role=LegRole.HEDGE)
    leg = conn.ledger.positions[pid]
    assert leg.role == LegRole.HEDGE
    assert leg.size == pytest.approx(3.0)
    assert conn.ledger.net_size() == pytest.approx(-2.0)
