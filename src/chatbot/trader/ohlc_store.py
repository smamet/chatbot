from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("open", "high", "low", "close")

# evenor: Date,Open,High,Low,Close[,Volume] (comma + header)
# backtestmarket: DD/MM/YYYY;HH:MM:SS;O;H;L;C;V (no header, GMT-6)
OHLC_SOURCES = ("evenor", "backtestmarket")
BACKTESTMARKET_TZ = "Etc/GMT+6"  # POSIX: GMT+6 == UTC-6


def load_ohlc_csv(
    path: Path,
    *,
    timezone: str = "Europe/Paris",
    source: str = "evenor",
) -> pd.DataFrame:
    """Load OHLCV CSV and normalize to a tz-aware index in `timezone`."""
    src = (source or "evenor").strip().lower()
    if src not in OHLC_SOURCES:
        raise ValueError(f"Unknown OHLC source '{source}'. Expected one of: {', '.join(OHLC_SOURCES)}")
    if src == "backtestmarket":
        df = _load_backtestmarket_csv(path)
    else:
        df = _load_evenor_csv(path)
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(timezone)
    keep = list(REQUIRED) + (["volume"] if "volume" in df.columns else [])
    return df[keep].astype(float)


def _load_evenor_csv(path: Path) -> pd.DataFrame:
    """Date,Open,High,Low,Close[,Volume] — comma-separated with header."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for need in ("date", "datetime", "time", "timestamp"):
        if need in cols:
            rename[cols[need]] = "ts"
            break
    for need in REQUIRED + ("volume",):
        if need in cols:
            rename[cols[need]] = need
    df = df.rename(columns=rename)
    if "ts" not in df.columns:
        raise ValueError(f"CSV missing datetime column: {path}")
    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"CSV missing {col}: {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def _load_backtestmarket_csv(path: Path) -> pd.DataFrame:
    """
    BacktestMarket MX 15m style:
    DD/MM/YYYY;HH:MM:SS;Open;High;Low;Close;Volume — no header, timezone GMT-6.
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        engine="python",
    )
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"CSV missing {col}: {path}")
    combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    ts = pd.to_datetime(combined, dayfirst=True, errors="coerce")
    # Vendor documents timezone as GMT-6.
    ts = ts.dt.tz_localize(BACKTESTMARKET_TZ, nonexistent="shift_forward", ambiguous="NaT")
    df = df.copy()
    df["ts"] = ts
    return df


# Relative to the last bar in the dataset.
BACKTEST_PERIODS: dict[str, str] = {
    "1w": "1 week",
    "2w": "2 weeks",
    "1m": "1 month",
    "3m": "3 months",
    "6m": "6 months",
    "1y": "1 year",
    "all": "All",
}


def period_bounds(df: pd.DataFrame, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive (start, end) timestamps for a backtest period relative to last bar."""
    if df.empty:
        raise ValueError("OHLC dataframe is empty")
    end = df.index.max()
    key = (period or "all").strip().lower()
    if key in ("", "all"):
        return df.index.min(), end
    if key == "1w":
        start = end - pd.Timedelta(weeks=1)
    elif key == "2w":
        start = end - pd.Timedelta(weeks=2)
    elif key == "1m":
        start = end - pd.DateOffset(months=1)
    elif key == "3m":
        start = end - pd.DateOffset(months=3)
    elif key == "6m":
        start = end - pd.DateOffset(months=6)
    elif key == "1y":
        start = end - pd.DateOffset(years=1)
    else:
        raise ValueError(
            f"Unknown backtest period '{period}'. Expected one of: {', '.join(BACKTEST_PERIODS)}"
        )
    return pd.Timestamp(start), pd.Timestamp(end)


def slice_ohlc_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Keep bars from (end - period) through end. `all` returns df unchanged."""
    key = (period or "all").strip().lower()
    if key in ("", "all") or df.empty:
        return df
    start, end = period_bounds(df, key)
    sliced = df.loc[start:end]
    if sliced.empty:
        raise ValueError(f"No bars in selected period '{key}' (dataset ends {end})")
    return sliced


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open", "close"])
    return out


def window_asof(df: pd.DataFrame, ts: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """Bars with index <= ts, last lookback rows (no lookahead)."""
    if ts.tzinfo is None and df.index.tz is not None:
        ts = ts.tz_localize(df.index.tz)
    elif ts.tzinfo is not None and df.index.tz is not None:
        ts = ts.tz_convert(df.index.tz)
    sliced = df.loc[:ts]
    return sliced.iloc[-lookback:]


# 15m series: contiguous when successive stamps differ by exactly one slot.
OHLC_15M_DELTA = pd.Timedelta(minutes=15)
# Gaps older than this are "history noise" for live (vendor CSV holes).
RECENT_GAP_DAYS = 90
# Cap dealing-hour probes so huge weekend/holiday holes stay cheap.
_MAX_GAP_PROBES = 96 * 14  # ~2 weeks of 15m slots


def next_15m_ts(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts) + OHLC_15M_DELTA


def connects_15m(prev: pd.Timestamp, nxt: pd.Timestamp) -> bool:
    """True when ``nxt`` is the immediate next 15m bar after ``prev`` (or equal)."""
    a = pd.Timestamp(prev)
    b = pd.Timestamp(nxt)
    return b <= next_15m_ts(a)


def _as_aware_pair(
    prev: pd.Timestamp, nxt: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(prev)
    b = pd.Timestamp(nxt)
    if a.tzinfo is not None and b.tzinfo is not None and a.tzinfo != b.tzinfo:
        b = b.tz_convert(a.tzinfo)
    return a, b


def _gap_contains_closed_dealing(
    prev: pd.Timestamp,
    nxt: pd.Timestamp,
    *,
    calendar_id: str | None,
    bar_delta: pd.Timedelta,
) -> bool:
    """True when at least one missing 15m slot falls outside the market calendar."""
    from chatbot.trader.market_calendar import is_dealing_open, resolve_calendar_id

    cal = resolve_calendar_id(calendar_id=calendar_id)
    probe = pd.Timestamp(prev) + bar_delta
    end = pd.Timestamp(nxt)
    steps = 0
    while probe < end:
        dt = probe.to_pydatetime()
        if not is_dealing_open(dt, calendar_id=cal):
            return True
        probe = probe + bar_delta
        steps += 1
        if steps > _MAX_GAP_PROBES:
            return True
    return False


def _cash_session_pause_break(prev: pd.Timestamp, nxt: pd.Timestamp, cash: tuple[int, int]) -> bool:
    """
    Index CFD overnight ↔ cash open quirks (Paris-local hours).

    Same calendar day inside cash hours ⇒ not natural.
    Pre-open tails (e.g. last=06:15 → next=08:00/09:00) are natural.
    """
    start_h, end_h = cash
    a, b = _as_aware_pair(prev, nxt)
    if a.date() != b.date():
        return True
    a_out = a.hour < start_h or a.hour >= end_h
    b_out = b.hour < start_h or b.hour >= end_h
    if a_out and b_out:
        return True
    if a.hour < start_h and not b_out:
        return b.hour <= start_h + 1
    if a.hour >= end_h or b.hour >= end_h:
        return True
    return False


def is_natural_session_break(
    prev: pd.Timestamp,
    nxt: pd.Timestamp,
    *,
    calendar_id: str | None = None,
    bar_delta: pd.Timedelta = OHLC_15M_DELTA,
) -> bool:
    """
    Overnight / weekend / holiday-style hole (not a mid-session missing candle).

    Epic-agnostic: uses the market calendar's dealing window. Index cash calendars
    also treat overnight↔cash pauses as natural; FX 24×5 does not (mid-week holes
    stay real gaps). Charts themselves never invent filler candles for these holes.
    """
    from chatbot.trader.market_calendar import cash_session_hours, resolve_calendar_id

    a, b = _as_aware_pair(prev, nxt)
    if b <= a + bar_delta:
        return True
    cal_id = resolve_calendar_id(calendar_id=calendar_id)
    if _gap_contains_closed_dealing(a, b, calendar_id=cal_id, bar_delta=bar_delta):
        return True
    cash = cash_session_hours(cal_id)
    if cash is not None:
        return _cash_session_pause_break(a, b, cash)
    return False


def find_intrasession_gaps(
    df: pd.DataFrame,
    *,
    bar_delta: pd.Timedelta = OHLC_15M_DELTA,
    calendar_id: str | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]:
    """Return mid-session holes (prev, next, delta) in a 15m OHLC frame."""
    if df is None or len(df) < 2:
        return []
    idx = df.index.sort_values()
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]] = []
    for a, b in zip(idx[:-1], idx[1:]):
        delta = b - a
        if delta <= bar_delta:
            continue
        if is_natural_session_break(a, b, calendar_id=calendar_id, bar_delta=bar_delta):
            continue
        gaps.append((pd.Timestamp(a), pd.Timestamp(b), delta))
    return gaps


def _gap_sample(
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]],
    *,
    max_samples: int,
    bar_delta: pd.Timedelta,
) -> list[dict]:
    samples: list[dict] = []
    for a, b, delta in gaps[: max(0, int(max_samples))]:
        missing = max(0, int(delta / bar_delta) - 1)
        samples.append(
            {
                "from": str(a),
                "to": str(b),
                "delta": str(delta),
                "missing_bars_approx": missing,
            }
        )
    return samples


def summarize_ohlc_gaps(
    df: pd.DataFrame,
    *,
    max_samples: int = 8,
    bar_delta: pd.Timedelta = OHLC_15M_DELTA,
    recent_days: int = RECENT_GAP_DAYS,
    calendar_id: str | None = None,
) -> dict:
    """
    UI-friendly mid-session gap report + how to fix.

    Overnight / weekend holes are ignored. Gaps older than ``recent_days``
    are reported as historical vendor noise (not a live blocker).
    """
    gaps = find_intrasession_gaps(df, bar_delta=bar_delta, calendar_id=calendar_id)
    empty = {
        "has_gaps": False,
        "has_recent_gaps": False,
        "gap_count": 0,
        "recent_gap_count": 0,
        "historical_gap_count": 0,
        "gaps": [],
        "fix_hint": None,
        "fix_steps": [],
        "severity": "ok",
    }
    if not gaps:
        return empty

    last_bar = pd.Timestamp(df.index.max())
    tip_horizon = last_bar - pd.Timedelta(days=max(1, int(recent_days)))
    recent = [g for g in gaps if g[1] >= tip_horizon]
    historical = [g for g in gaps if g[1] < tip_horizon]
    recent_count = len(recent)
    hist_count = len(historical)
    total = len(gaps)

    if recent_count:
        samples = _gap_sample(recent, max_samples=max_samples, bar_delta=bar_delta)
        fix_hint = (
            f"{recent_count} mid-session hole(s) in the last {recent_days} days "
            "(affects live charts / Gemini). Fill with Sync from IG, or re-upload "
            "CSV if Sync cannot cover the hole."
        )
        steps = [
            "Click Sync from IG (needs allowance) to fill from the last contiguous candle.",
            "Refresh — recent gap count should be 0.",
            "If Sync fails: Upload a clean BacktestMarket CSV, then Sync again.",
        ]
        return {
            "has_gaps": True,
            "has_recent_gaps": True,
            "gap_count": total,
            "recent_gap_count": recent_count,
            "historical_gap_count": hist_count,
            "gaps": samples,
            "truncated": recent_count > len(samples),
            "fix_hint": fix_hint,
            "fix_steps": steps,
            "last_gap_to": str(recent[-1][1]),
            "severity": "error",
            "recent_days": recent_days,
        }

    # History-only holes (common in long BacktestMarket files) — not a live issue.
    samples = _gap_sample(historical[-max_samples:], max_samples=max_samples, bar_delta=bar_delta)
    return {
        "has_gaps": False,  # not a live blocker
        "has_recent_gaps": False,
        "gap_count": total,
        "recent_gap_count": 0,
        "historical_gap_count": hist_count,
        "gaps": samples,
        "truncated": hist_count > len(samples),
        "fix_hint": (
            f"{hist_count} mid-session hole(s) only in older history "
            f"(before last {recent_days} days). Live lookback is clean — no action needed "
            "for Paper/Live. Optional: re-upload a cleaner CSV if you backtest across those years."
        ),
        "fix_steps": [],
        "last_gap_to": str(historical[-1][1]) if historical else None,
        "severity": "info",
        "recent_days": recent_days,
    }


def assert_append_contiguous(
    last_ts: pd.Timestamp | None,
    fresh: pd.DataFrame,
    *,
    allow_session_breaks: bool = True,
    calendar_id: str | None = None,
) -> pd.DataFrame:
    """
    Keep only bars after ``last_ts`` and refuse discontinuous mid-session splices.

    Raises ValueError if appending would create an intrasession hole
    (e.g. last=10:00 then first new=11:00 on the same day).
    """
    if fresh is None or fresh.empty:
        return fresh
    out = fresh.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if last_ts is None:
        return out
    last = pd.Timestamp(last_ts)
    if out.index.tz is not None:
        if last.tzinfo is None:
            last = last.tz_localize(out.index.tz)
        else:
            last = last.tz_convert(out.index.tz)
    out = out.loc[out.index > last]
    if out.empty:
        return out
    first = pd.Timestamp(out.index[0])
    if connects_15m(last, first):
        return out
    if allow_session_breaks and is_natural_session_break(
        last, first, calendar_id=calendar_id
    ):
        return out
    raise ValueError(
        f"Refusing discontinuous OHLC append: last={last} → first_new={first} "
        f"(would leave a mid-session hole). Sync/range-fill the missing bars first."
    )


def append_bars(
    path: Path,
    df: pd.DataFrame,
    *,
    require_contiguous: bool = False,
    calendar_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_ohlc_csv(path, timezone=str(df.index.tz) if df.index.tz else "UTC")
        if require_contiguous and not existing.empty:
            df = assert_append_contiguous(
                pd.Timestamp(existing.index[-1]), df, calendar_id=calendar_id
            )
        merged = pd.concat([existing, df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = df
    out = merged.reset_index()
    out.rename(columns={"ts": "Date", "index": "Date"}, inplace=True)
    if "Date" not in out.columns:
        out.columns = ["Date"] + list(out.columns[1:])
    # normalize column names for export
    mapping = {
        out.columns[0]: "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = out.rename(columns={k: v for k, v in mapping.items() if k in out.columns})
    out.to_csv(path, index=False)
