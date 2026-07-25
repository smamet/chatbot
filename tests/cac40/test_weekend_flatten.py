"""Weekend / holiday flatten window + session idle + auto-hedge protection."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from chatbot.application.cac40_live_service import (
    _load_market_closed_groups,
    merge_decisions_with_market_closed,
)
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.llm_decision import build_user_payload
from chatbot.cac40.market_calendar import (
    euronext_closures,
    flatten_check,
    is_dealing_open,
    is_trading_day,
    session_snapshot,
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


LONDON = ZoneInfo("Europe/London")
PARIS = ZoneInfo("Europe/Paris")


def _open_session(**overrides):
    base = {
        "dealing_open": True,
        "now": "2026-07-24T12:00:00+01:00",
        "weekday": "Friday",
        "tz": "Europe/London",
        "source": "test",
        "weekly_open": "Sun 23:02 Europe/London",
        "weekly_close": "Fri 22:00 Europe/London",
        "next_open": None,
        "next_close": "2026-07-24T22:00:00+01:00",
        "close_id": None,
        "flatten_enabled": True,
        "flatten_now": False,
        "flatten_reason": "",
        "flatten_reasons": [],
        "flatten_close_at": "2026-07-24T22:00:00+01:00",
        "flatten_window_start": "2026-07-24T21:30:00+01:00",
        "minutes_to_close": 600,
        "next_open_day": "2026-07-27",
    }
    base.update(overrides)
    return base


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


def test_is_dealing_open_weekly_window():
    # Friday before close
    assert is_dealing_open(datetime(2026, 7, 24, 21, 59, tzinfo=LONDON)) is True
    # Friday at close
    assert is_dealing_open(datetime(2026, 7, 24, 22, 0, tzinfo=LONDON)) is False
    # Saturday
    assert is_dealing_open(datetime(2026, 7, 25, 12, 0, tzinfo=LONDON)) is False
    # Sunday before open
    assert is_dealing_open(datetime(2026, 7, 26, 23, 0, tzinfo=LONDON)) is False
    # Sunday at open
    assert is_dealing_open(datetime(2026, 7, 26, 23, 2, tzinfo=LONDON)) is True
    # Mid-week overnight
    assert is_dealing_open(datetime(2026, 7, 22, 2, 0, tzinfo=LONDON)) is True


def test_is_dealing_open_holiday_and_eve():
    # Labour Day Friday 2026-05-01 — closed all day
    assert is_dealing_open(datetime(2026, 5, 1, 12, 0, tzinfo=LONDON)) is False
    # Thursday eve before Labour Day — closed from 22:00 London
    assert is_dealing_open(datetime(2026, 4, 30, 21, 59, tzinfo=LONDON)) is True
    assert is_dealing_open(datetime(2026, 4, 30, 22, 0, tzinfo=LONDON)) is False
    # Christmas 2026 is Friday — closed
    assert is_dealing_open(datetime(2026, 12, 25, 10, 0, tzinfo=LONDON)) is False


def test_flatten_window_friday_lead_30_london():
    # Friday 2026-07-24 21:29 London — inactive
    assert flatten_check(
        datetime(2026, 7, 24, 21, 29, tzinfo=LONDON),
        lead_minutes=30,
    )["active"] is False
    # 21:30 — active (weekend), still dealing
    fri = flatten_check(
        datetime(2026, 7, 24, 21, 30, tzinfo=LONDON),
        lead_minutes=30,
    )
    assert fri["active"] is True
    assert fri["dealing_open"] is True
    assert "weekend" in fri["reason"]
    # After close — inactive (idle branch takes over)
    after = flatten_check(
        datetime(2026, 7, 24, 22, 1, tzinfo=LONDON),
        lead_minutes=30,
    )
    assert after["active"] is False
    assert after["dealing_open"] is False


def test_flatten_window_thursday_before_friday_holiday():
    # Labour Day 2026 is Friday May 1 → flatten Thursday Apr 30 evening London
    check = flatten_check(
        datetime(2026, 4, 30, 21, 45, tzinfo=LONDON),
        lead_minutes=30,
    )
    assert check["active"] is True
    assert "holiday: Labour Day" in check["reason"]


def test_flatten_friday_before_easter_monday_reports_weekend_and_holiday():
    # Good Friday 2026-04-03 — flatten Thursday 2026-04-02.
    check = flatten_check(
        datetime(2026, 4, 2, 21, 40, tzinfo=LONDON),
        lead_minutes=30,
    )
    assert check["active"] is True
    assert "holiday: Good Friday" in check["reason"]


def test_session_snapshot_resume_fields():
    closed = session_snapshot(datetime(2026, 7, 25, 12, 0, tzinfo=LONDON))
    assert closed["dealing_open"] is False
    assert closed["next_open"] is not None
    assert "2026-07-26T23:02" in closed["next_open"]
    assert closed["close_id"]

    open_snap = session_snapshot(datetime(2026, 7, 26, 23, 5, tzinfo=LONDON))
    assert open_snap["dealing_open"] is True
    assert open_snap["next_open"] is None
    assert open_snap["next_close"] is not None


def test_payload_includes_market_clock_and_flatten_instruction():
    snap = MarketSnapshot(symbol="CAC40", last_price=7500.0, phase="LongAtSupport")
    clock = {
        "flatten_now": True,
        "reason": "weekend",
        "net_exposure": 2.0,
        "weekday": "Friday",
        "now": "2026-07-24T21:45:00+01:00",
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

    monkeypatch.setattr(
        "chatbot.cac40.scheduler.session_snapshot",
        lambda *a, **k: _open_session(
            flatten_now=True,
            flatten_reason="weekend",
            flatten_reasons=["weekend"],
            minutes_to_close=15,
            now="2026-07-24T21:45:00+01:00",
        ),
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
        "chatbot.cac40.scheduler.session_snapshot",
        lambda *a, **k: _open_session(
            flatten_now=True,
            flatten_reason="weekend",
            flatten_reasons=["weekend"],
            minutes_to_close=10,
        ),
    )
    payload = sched.run_once()
    assert payload["auto_flatten"] is None or payload["auto_flatten"].get("hedged") is False
    assert sched.ig.ledger.net_size() == pytest.approx(0.0)
    assert len(sched.ig.ledger.positions) == 2


def test_idle_when_market_closed_skips_ohlc_and_llm(tmp_path: Path, monkeypatch):
    cfg = Cac40Config(chart_show_pivots=False, chart_show_rsi=False)
    sched = LiveScheduler(cfg, api_key="", journal_dir=tmp_path, dry_run=True)
    called = {"ohlc": False, "login": False}

    def boom_ohlc():
        called["ohlc"] = True
        raise AssertionError("should not load OHLC while closed")

    def boom_login():
        called["login"] = True
        raise AssertionError("should not login while closed")

    sched.ohlc_provider = boom_ohlc
    sched.ensure_logged_in = boom_login  # type: ignore[method-assign]
    sched.llm.decide = lambda *a, **k: (_ for _ in ()).throw(AssertionError("llm"))

    monkeypatch.setattr(
        "chatbot.cac40.scheduler.session_snapshot",
        lambda *a, **k: {
            **_open_session(
                dealing_open=False,
                flatten_now=False,
                next_open="2026-07-26T23:02:00+01:00",
                close_id="20260724_2200",
            ),
            "dealing_open": False,
        },
    )
    payload = sched.run_once()
    assert payload["skipped"] is True
    assert payload["skip_reason"] == "market_closed"
    assert payload["mirror"] == []
    assert payload["executed"] == []
    assert not called["ohlc"]
    assert not called["login"]
    hb = tmp_path / "market_closed" / "20260724_2200.jsonl"
    assert hb.is_file()
    assert "market_closed" in hb.read_text(encoding="utf-8")
    # No per-skip cycle dirs
    assert not any(p.is_dir() and p.name[:8].isdigit() for p in tmp_path.iterdir())


def test_resume_when_open_runs_full_path(tmp_path: Path, monkeypatch):
    cfg = Cac40Config(
        chart_show_pivots=False,
        chart_show_rsi=False,
        llm_trigger_mode="interval",
        llm_every_bars=999,
    )
    sched = LiveScheduler(cfg, api_key="", journal_dir=tmp_path, dry_run=True)
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
            "skip_llm": False,
            "warnings": [],
            "error": None,
            "allowance": None,
            "meta": {"source": "local_csv"},
        },
    )()
    sched.ig._cst = "cst"  # noqa: SLF001
    sched.fm.notify = lambda *a, **k: None
    quiet = type("T", (), {"should_call": False, "reasons": ["quiet"]})()
    sched.trigger.evaluate = lambda **k: quiet  # type: ignore[method-assign]

    monkeypatch.setattr(
        "chatbot.cac40.scheduler.session_snapshot",
        lambda *a, **k: _open_session(),
    )
    payload = sched.run_once(force_llm=False)
    assert payload.get("skip_reason") != "market_closed"
    assert payload.get("cycle_dir")
    assert (tmp_path / payload["cycle_dir"] / "cycle.json").is_file()


def test_market_closed_groups_many_heartbeats(tmp_path: Path):
    root = tmp_path / "market_closed"
    root.mkdir(parents=True)
    path = root / "20260724_2200.jsonl"
    lines = [
        '{"ts": "2026-07-24T21:05:00+00:00", "next_open": "2026-07-26T22:02:00+00:00", "skip_reason": "market_closed"}',
        '{"ts": "2026-07-24T21:20:00+00:00", "next_open": "2026-07-26T22:02:00+00:00", "skip_reason": "market_closed"}',
        '{"ts": "2026-07-24T21:35:00+00:00", "next_open": "2026-07-26T22:02:00+00:00", "skip_reason": "market_closed"}',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = _load_market_closed_groups(tmp_path)
    assert len(groups) == 1
    g = groups[0]
    assert g["kind"] == "market_closed_group"
    assert g["heartbeat_count"] == 3
    assert len(g["heartbeats"]) == 3
    decisions = [{"ts": "2026-07-24T20:00:00+00:00", "bias": "long"}]
    merged = merge_decisions_with_market_closed(decisions, groups)
    assert merged[0]["kind"] == "market_closed_group"
    assert merged[1]["bias"] == "long"


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
