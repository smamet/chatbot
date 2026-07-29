"""Local CSV-backed OHLC feed for live cycles (minimizes IG historical allowance)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from chatbot.trader.chart_renderer import pivot_history_pad
from chatbot.trader.config import TraderConfig
from chatbot.trader.ig_connector import IgApiError, IgConnector
from chatbot.trader.ohlc_store import (
    append_bars,
    assert_append_contiguous,
    connects_15m,
    find_intrasession_gaps,
    is_natural_session_break,
    load_ohlc_csv,
    next_15m_ts,
    resample_ohlc,
    window_asof,
)

logger = logging.getLogger(__name__)

# Cheap path: latest N closed 15m bars when the CSV is only slightly behind.
LIVE_TOP_UP_BARS = 8
# Proceed with cache if last bar is within this many 15m slots of "now".
STALE_OK_SLOTS = 2
# Match manual Sync: refuse automatic catch-up beyond this gap (re-upload instead).
LIVE_MAX_GAP_DAYS = 60


@dataclass
class LiveOhlcFeed:
    """Frames + metadata for one live cycle."""

    ohlc_15: pd.DataFrame
    ohlc_1h: pd.DataFrame
    ohlc_1d: pd.DataFrame
    last_price: float
    last_bar_ts: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    top_up_added: int = 0
    top_up_ok: bool = True
    stale: bool = False
    skip_llm: bool = False
    allowance: dict[str, Any] | None = None


def expected_last_closed_15m(
    now: datetime | None = None, *, tz: str = "Europe/Paris"
) -> pd.Timestamp:
    """Most recent *closed* 15m bar timestamp in ``tz``."""
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    local = current.tz_convert(tz)
    floored = local.floor("15min")
    # During the open bar, last closed is previous slot; at exact close, previous.
    closed = floored - pd.Timedelta(minutes=15)
    return closed


def _bar_age_slots(last_ts: pd.Timestamp, now: datetime | None = None) -> float:
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    last = pd.Timestamp(last_ts)
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    delta = current - last
    return float(delta.total_seconds() / (15 * 60))


def _clock_end(now: datetime | None, timezone_name: str) -> pd.Timestamp:
    end = pd.Timestamp(now or datetime.now(timezone.utc))
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    return end.tz_convert(timezone_name)


def _fetch_range(
    connector: IgConnector,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone_name: str,
) -> pd.DataFrame:
    from chatbot.trader.ig_ohlc import catchup_ohlc_15m

    df, mode = catchup_ohlc_15m(
        connector, start=start, end=end, timezone=timezone_name
    )
    logger.info("Live OHLC catch-up mode=%s bars=%s", mode, 0 if df is None else len(df))
    return df if df is not None else pd.DataFrame()


def _prepare_newer_bars(
    fresh: pd.DataFrame,
    *,
    last_local: pd.Timestamp,
    timezone_name: str,
) -> pd.DataFrame:
    if fresh is None or fresh.empty:
        return fresh.iloc[0:0] if fresh is not None else pd.DataFrame()
    df = fresh.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(timezone_name)
    return assert_append_contiguous(last_local, df, allow_session_breaks=True)


def top_up_csv_from_connector(
    path: Path,
    connector: IgConnector,
    *,
    max_bars: int = LIVE_TOP_UP_BARS,
    timezone_name: str = "Europe/Paris",
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Append missing 15m bars since CSV last ts — never creating mid-session holes.

    - Small lag: try cheap ``get_ohlc(15m, N)`` only if those bars connect to
      ``last_ts + 15m`` (or a natural session break).
    - Otherwise (or if cheap would splice a hole): paged range from
      ``last_ts + 15m`` → now so every missing candle is filled.

    Returns ``{added, allowance, last_candle, mode}``.
    """
    existing = load_ohlc_csv(path, timezone=timezone_name)
    if existing.empty:
        raise ValueError(f"OHLC CSV is empty: {path}")
    last_ts = pd.Timestamp(existing.index[-1])
    expected = expected_last_closed_15m(now=now, tz=timezone_name)
    last_local = (
        last_ts.tz_convert(timezone_name)
        if last_ts.tzinfo
        else last_ts.tz_localize(timezone_name)
    )
    if last_local >= expected:
        return {
            "added": 0,
            "allowance": getattr(connector, "last_price_allowance", None),
            "last_candle": str(last_ts),
            "skipped_fetch": True,
            "mode": "skip",
        }

    gap = expected - last_local
    if gap > pd.Timedelta(days=LIVE_MAX_GAP_DAYS):
        raise ValueError(
            f"OHLC gap is {gap.days} days (max {LIVE_MAX_GAP_DAYS}). "
            "Re-upload a CSV, then Sync from IG."
        )

    end = _clock_end(now, timezone_name)
    # Inclusive of last bar — IG from/to is flaky if we start exactly at last+15m.
    start = last_local
    cheap_window = pd.Timedelta(minutes=15 * max(1, int(max_bars)))
    mode = "cheap"
    fresh: pd.DataFrame | None = None

    use_range = gap > cheap_window
    if not use_range:
        tip = connector.get_ohlc("15m", max(1, int(max_bars)))
        if tip is not None and not tip.empty:
            if tip.index.tz is None:
                tip.index = tip.index.tz_localize("UTC")
            tip.index = tip.index.tz_convert(timezone_name)
            tip_newer = tip.loc[tip.index > last_local]
            if tip_newer.empty:
                # Tip has nothing newer — try range so we don't silently stay behind.
                use_range = True
                logger.info("Live OHLC cheap tip empty/not-newer; range-filling")
            else:
                first = pd.Timestamp(tip_newer.index[0])
                if connects_15m(last_local, first) or is_natural_session_break(
                    last_local, first
                ):
                    fresh = tip_newer
                else:
                    # Cheap tip would skip bars (e.g. last=10:00, tip starts 11:00).
                    use_range = True
                    logger.info(
                        "Live OHLC cheap tip discontinuous (%s → %s); range-filling",
                        last_local,
                        first,
                    )
        else:
            use_range = True
            logger.info("Live OHLC cheap tip empty; range-filling")

    if use_range:
        mode = "range"
        fresh = _fetch_range(
            connector, start=start, end=end, timezone_name=timezone_name
        )

    allowance = getattr(connector, "last_price_allowance", None)
    if fresh is None or fresh.empty:
        return {
            "added": 0,
            "allowance": allowance,
            "last_candle": str(last_ts),
            "skipped_fetch": False,
            "mode": mode,
        }

    newer = _prepare_newer_bars(fresh, last_local=last_local, timezone_name=timezone_name)
    added = 0
    if not newer.empty:
        # Validate the stretch *before* writing — never corrupt the CSV.
        stretch_gaps = find_intrasession_gaps(newer)
        if stretch_gaps:
            a, b, delta = stretch_gaps[0]
            raise ValueError(
                f"IG returned OHLC with mid-session hole {a} → {b} ({delta}); "
                "refusing to corrupt local CSV"
            )
        append_bars(path, newer, require_contiguous=True)
        added = int(len(newer))
        last_ts = pd.Timestamp(newer.index[-1])
    return {
        "added": added,
        "allowance": allowance,
        "last_candle": str(last_ts),
        "skipped_fetch": False,
        "mode": mode,
    }


def build_live_frames(
    df_15: pd.DataFrame,
    config: TraderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Window 15m/1H/1D frames (same sizes as the old IG lookbacks)."""
    if df_15.empty:
        empty = df_15.iloc[0:0].copy()
        return empty, empty.copy(), empty.copy()
    ts = pd.Timestamp(df_15.index[-1])
    rsi_seed = max(2, int(config.warmup_bars or 14))
    pivots_on = bool(config.chart_show_pivots)
    pivot_period = config.chart_pivot_period or "D"
    pad_15 = pivot_history_pad(pivot_period, timeframe="15m") if pivots_on else 0
    pad_1h = pivot_history_pad(pivot_period, timeframe="1h") if pivots_on else 0
    df_1h = resample_ohlc(df_15, "1h")
    df_1d = resample_ohlc(df_15, "1D")
    w15 = window_asof(df_15, ts, int(config.lookback_15m) + rsi_seed + pad_15)
    w1h = window_asof(df_1h, ts, int(config.lookback_1h) + rsi_seed + pad_1h)
    w1d = window_asof(df_1d, ts, int(config.lookback_1d) + rsi_seed)
    return w15, w1h, w1d


def prepare_live_ohlc_feed(
    path: Path,
    *,
    config: TraderConfig,
    connector: IgConnector | None = None,
    top_up: bool = True,
    now: datetime | None = None,
    stream_healthy: bool | None = None,
    stream_stale: bool = False,
    stream_mid: float | None = None,
    stream_error: str | None = None,
) -> LiveOhlcFeed:
    """
    Load local 15m CSV, optionally top up bars from IG, resample higher TFs.

    Never bootstraps an empty CSV (caller must Sync/upload once).
    Skips LLM when the chart window still contains mid-session holes.

    When ``stream_healthy`` is True, skips hot-path REST ``/prices`` top-up
    unless the CSV is behind the expected closed bar (one gap-repair fetch).
    When ``stream_stale`` is True during an open session, fail-closed for LLM.
    """
    tz = str(config.data_timezone or "Europe/Paris")
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not path.exists() or path.stat().st_size <= 0:
        return LiveOhlcFeed(
            ohlc_15=empty,
            ohlc_1h=empty.copy(),
            ohlc_1d=empty.copy(),
            last_price=0.0,
            error="OHLC CSV missing — upload history or Sync from IG once",
            top_up_ok=False,
            skip_llm=True,
        )

    warnings: list[str] = []
    allowance: dict[str, Any] | None = None
    top_up_added = 0
    top_up_ok = True
    stale = False
    skip_llm = False
    error: str | None = None

    # Stream-aware top-up: healthy → skip /prices unless CSV behind expected bar.
    do_top_up = bool(top_up and connector is not None)
    if stream_stale:
        do_top_up = False
        stale = True
        skip_llm = True
        msg = stream_error or "Lightstreamer OHLC stream is stale — skipping LLM"
        error = msg
        warnings.append(msg)
    elif stream_healthy:
        do_top_up = False
        try:
            peek = load_ohlc_csv(path, timezone=tz)
            if not peek.empty:
                last_ts = pd.Timestamp(peek.index[-1])
                last_local = (
                    last_ts.tz_convert(tz) if last_ts.tzinfo else last_ts.tz_localize(tz)
                )
                expected = expected_last_closed_15m(now=now, tz=tz)
                if last_local < expected and not is_natural_session_break(
                    last_local, expected
                ):
                    do_top_up = True
                    warnings.append(
                        "stream_gap_repair: CSV behind expected closed bar — one REST top-up"
                    )
        except Exception as exc:
            warnings.append(f"stream_gap_check_failed: {exc}")
            do_top_up = bool(top_up and connector is not None)

    if do_top_up and connector is not None:
        try:
            result = top_up_csv_from_connector(
                path,
                connector,
                max_bars=LIVE_TOP_UP_BARS,
                timezone_name=tz,
                now=now,
            )
            top_up_added = int(result.get("added") or 0)
            if isinstance(result.get("allowance"), dict):
                allowance = result["allowance"]
            mode = result.get("mode")
            if mode == "range" and top_up_added:
                logger.info(
                    "Live OHLC range catch-up added %s bars (contiguous fill)",
                    top_up_added,
                )
        except IgApiError as exc:
            top_up_ok = False
            msg = str(exc)
            logger.warning("Live OHLC top-up failed: %s", msg)
            warnings.append(f"ohlc_top_up_failed: {msg.splitlines()[0]}")
        except Exception as exc:
            top_up_ok = False
            logger.exception("Live OHLC top-up failed")
            warnings.append(f"ohlc_top_up_failed: {exc}")

    try:
        df_full = load_ohlc_csv(path, timezone=tz)
    except Exception as exc:
        return LiveOhlcFeed(
            ohlc_15=empty,
            ohlc_1h=empty.copy(),
            ohlc_1d=empty.copy(),
            last_price=0.0,
            error=f"Failed to load OHLC CSV: {exc}",
            top_up_ok=False,
            skip_llm=True,
            warnings=warnings,
            allowance=allowance,
        )

    if df_full.empty:
        return LiveOhlcFeed(
            ohlc_15=empty,
            ohlc_1h=empty.copy(),
            ohlc_1d=empty.copy(),
            last_price=0.0,
            error="OHLC CSV is empty — upload history or Sync from IG once",
            top_up_ok=False,
            skip_llm=True,
            warnings=warnings,
            allowance=allowance,
        )

    last_ts = pd.Timestamp(df_full.index[-1])
    last_local = (
        last_ts.tz_convert(tz) if last_ts.tzinfo else last_ts.tz_localize(tz)
    )
    expected = expected_last_closed_15m(now=now, tz=tz)
    age_slots = _bar_age_slots(last_ts, now=now)
    slots_behind_expected = max(
        0.0, float((expected - last_local) / pd.Timedelta(minutes=15))
    )

    if not top_up_ok:
        if age_slots <= STALE_OK_SLOTS:
            stale = True
            warnings.append(
                f"stale_data: using cached bars (last={last_ts}, age≈{age_slots:.1f}×15m)"
            )
        else:
            stale = True
            skip_llm = True
            error = (
                f"OHLC top-up failed and cache is too old "
                f"(last={last_ts}, age≈{age_slots:.1f}×15m). "
                "Waiting for IG historical allowance or Sync."
            )
            warnings.append(error)
    elif (
        slots_behind_expected > STALE_OK_SLOTS
        and not is_natural_session_break(last_local, expected)
    ):
        # Top-up "succeeded" with 0 bars but we're still mid-session behind.
        stale = True
        skip_llm = True
        rem = None
        if isinstance(allowance, dict):
            rem = allowance.get("remaining")
            if rem is None:
                rem = allowance.get("remainingAllowance")
        try:
            rem_i = int(rem) if rem is not None else None
        except (TypeError, ValueError):
            rem_i = None
        allowance_hint = (
            "IG historical allowance may be exhausted on this account "
            "(shared across API keys)."
            if rem_i is not None and rem_i <= 0
            else (
                "IG returned no newer bars (DEMO delay, epic, or empty range) — "
                "not necessarily allowance; try Sync details / CSV upload."
            )
        )
        error = (
            f"OHLC still behind expected closed bar "
            f"(last={last_local}, expected={expected}, "
            f"behind≈{slots_behind_expected:.1f}×15m). "
            f"{allowance_hint} Skipping LLM."
        )
        warnings.append(error)

    w15, w1h, w1d = build_live_frames(df_full, config)
    chart_gaps = find_intrasession_gaps(w15)
    if chart_gaps:
        a, b, delta = chart_gaps[0]
        skip_llm = True
        gap_msg = (
            f"OHLC chart window has mid-session gap {a} → {b} ({delta}); "
            "skipping LLM until Sync fills missing 15m bars"
        )
        error = gap_msg if not error else f"{error}\n{gap_msg}"
        warnings.append(gap_msg)
        logger.error(gap_msg)

    last_price = float(df_full["close"].iloc[-1])
    if stream_mid is not None:
        try:
            mid_f = float(stream_mid)
            if mid_f > 0:
                last_price = mid_f
        except (TypeError, ValueError):
            pass
    return LiveOhlcFeed(
        ohlc_15=w15,
        ohlc_1h=w1h,
        ohlc_1d=w1d,
        last_price=last_price,
        last_bar_ts=str(last_ts),
        warnings=warnings,
        error=error,
        top_up_added=top_up_added,
        top_up_ok=top_up_ok,
        stale=stale,
        skip_llm=skip_llm,
        allowance=allowance,
    )


LiveOhlcProvider = Callable[[], LiveOhlcFeed]
