from __future__ import annotations

from chatbot.trader.llm_trigger import (
    TRIGGER_BOOK_CHANGE,
    TRIGGER_BOOTSTRAP,
    TRIGGER_INTERVAL,
    TRIGGER_RESISTANCE_APPROACH,
    TRIGGER_RESISTANCE_BREAK,
    TRIGGER_SUPPORT_APPROACH,
    TRIGGER_SUPPORT_BREAK,
    LlmTrigger,
    classify_resistance_zone,
    classify_support_zone,
)


def _bar(o: float, h: float, l: float, c: float) -> dict[str, float]:
    return {"open": o, "high": h, "low": l, "close": c}


S = 7000.0
R = 7100.0
BAND = 15.0


def test_classify_support_zones() -> None:
    assert classify_support_zone(_bar(7050, 7060, 7040, 7055), S, BAND) == "mid_range"
    assert classify_support_zone(_bar(7010, 7020, 7005, 7012), S, BAND) == "in_band"
    assert classify_support_zone(_bar(6990, 7000, 6980, 6985), S, BAND) == "beyond"


def test_classify_resistance_zones() -> None:
    assert classify_resistance_zone(_bar(7050, 7060, 7040, 7055), R, BAND) == "mid_range"
    assert classify_resistance_zone(_bar(7090, 7105, 7085, 7095), R, BAND) == "in_band"
    assert classify_resistance_zone(_bar(7110, 7125, 7105, 7120), R, BAND) == "beyond"


def test_bootstrap_when_levels_missing() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    d = t.evaluate(bar=mid, support=None, resistance=None)
    assert d.should_call is True
    assert TRIGGER_BOOTSTRAP in d.reasons

    d2 = t.evaluate(bar=mid, support=S, resistance=None)
    assert d2.should_call is True
    assert TRIGGER_BOOTSTRAP in d2.reasons


def test_mid_range_skips_with_levels() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)
    d = t.evaluate(bar=mid, support=S, resistance=R)
    assert d.should_call is False
    assert d.reasons == ()


def test_support_approach_and_debounce() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    approach = _bar(7010, 7020, 7005, 7010)
    d = t.evaluate(bar=approach, support=S, resistance=R)
    assert d.should_call is True
    assert TRIGGER_SUPPORT_APPROACH in d.reasons

    t.on_success(bar=approach, support=S, resistance=R)
    # Still in band → no re-fire
    d2 = t.evaluate(bar=approach, support=S, resistance=R)
    assert d2.should_call is False


def test_approach_then_break_still_fires() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    approach = _bar(7010, 7020, 7005, 7010)
    d = t.evaluate(bar=approach, support=S, resistance=R)
    assert TRIGGER_SUPPORT_APPROACH in d.reasons
    t.on_success(bar=approach, support=S, resistance=R)

    brk = _bar(6985, 6995, 6970, 6980)
    d2 = t.evaluate(bar=brk, support=S, resistance=R)
    assert d2.should_call is True
    assert TRIGGER_SUPPORT_BREAK in d2.reasons


def test_gap_through_band_fires_break_only() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    gap = _bar(6980, 6990, 6960, 6970)  # mid → beyond in one bar
    d = t.evaluate(bar=gap, support=S, resistance=R)
    assert d.should_call is True
    assert TRIGGER_SUPPORT_BREAK in d.reasons
    assert TRIGGER_SUPPORT_APPROACH not in d.reasons


def test_resistance_approach_and_break() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    approach = _bar(7090, 7105, 7088, 7095)
    d = t.evaluate(bar=approach, support=S, resistance=R)
    assert TRIGGER_RESISTANCE_APPROACH in d.reasons
    t.on_success(bar=approach, support=S, resistance=R)

    brk = _bar(7110, 7130, 7108, 7125)
    d2 = t.evaluate(bar=brk, support=S, resistance=R)
    assert TRIGGER_RESISTANCE_BREAK in d2.reasons


def test_book_change_forces_call_in_quiet_zone() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    t.note_fills([{"type": "open", "fill": 7010}])
    d = t.evaluate(bar=mid, support=S, resistance=R)
    assert d.should_call is True
    assert TRIGGER_BOOK_CHANGE in d.reasons

    t.on_success(bar=mid, support=S, resistance=R)
    d2 = t.evaluate(bar=mid, support=S, resistance=R)
    assert d2.should_call is False


def test_failure_keeps_pending_and_retries() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    approach = _bar(7010, 7020, 7005, 7010)
    d = t.evaluate(bar=approach, support=S, resistance=R)
    assert TRIGGER_SUPPORT_APPROACH in d.reasons
    t.on_failure()

    # Same bar geometry → still pending / transition
    d2 = t.evaluate(bar=approach, support=S, resistance=R)
    assert d2.should_call is True
    assert TRIGGER_SUPPORT_APPROACH in d2.reasons


def test_rearm_after_return_to_mid_range() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)

    approach = _bar(7010, 7020, 7005, 7010)
    assert t.evaluate(bar=approach, support=S, resistance=R).should_call is True
    t.on_success(bar=approach, support=S, resistance=R)

    # Back mid → passive re-arm, no call
    d_mid = t.evaluate(bar=mid, support=S, resistance=R)
    assert d_mid.should_call is False
    assert t.support_zone == "mid_range"

    # Approach again
    d2 = t.evaluate(bar=approach, support=S, resistance=R)
    assert d2.should_call is True
    assert TRIGGER_SUPPORT_APPROACH in d2.reasons


def test_interval_mode() -> None:
    t = LlmTrigger(band_points=BAND, mode="interval", every_bars=4)
    mid = _bar(7050, 7060, 7040, 7055)
    assert t.evaluate(bar=mid, support=S, resistance=R, bar_index=0).should_call is True
    assert TRIGGER_INTERVAL in t.evaluate(bar=mid, support=S, resistance=R, bar_index=0).reasons
    assert t.evaluate(bar=mid, support=S, resistance=R, bar_index=1).should_call is False
    assert t.evaluate(bar=mid, support=S, resistance=R, bar_index=4).should_call is True


def test_interval_wall_clock_respects_last_llm_at() -> None:
    from datetime import datetime, timedelta, timezone

    mid = _bar(7050, 7060, 7040, 7055)
    t = LlmTrigger(
        band_points=BAND,
        mode="interval",
        every_bars=4,  # 4 × 15m = 1 hour
        interval_clock="wall",
    )
    now = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    # Never called → first fire allowed (bar path / no last).
    assert t.interval_due(bar_index=0, now=now) is True
    t.mark_llm_called(now)
    # 30 minutes later — too soon.
    assert t.interval_due(bar_index=99, now=now + timedelta(minutes=30)) is False
    # 1 hour later — due.
    assert t.interval_due(bar_index=99, now=now + timedelta(hours=1)) is True
    d = t.evaluate(bar=mid, support=S, resistance=R, bar_index=99)
    # evaluate uses "now" internally; with last_llm_at=now and real clock, may vary.
    # Force via interval_due already covered; ensure evaluate still interval-gated:
    t.last_llm_at = datetime.now(timezone.utc)
    assert t.evaluate(bar=mid, support=S, resistance=R, bar_index=0).should_call is False


def test_new_levels_recompute_zones_without_spurious_fire() -> None:
    t = LlmTrigger(band_points=BAND, mode="levels")
    mid = _bar(7050, 7060, 7040, 7055)
    t.on_success(bar=mid, support=S, resistance=R)
    d = t.evaluate(bar=mid, support=7040.0, resistance=7060.0)
    assert TRIGGER_SUPPORT_APPROACH not in d.reasons
    assert TRIGGER_RESISTANCE_APPROACH not in d.reasons
