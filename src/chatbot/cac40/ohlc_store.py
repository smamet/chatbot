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


def append_bars(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_ohlc_csv(path, timezone=str(df.index.tz) if df.index.tz else "UTC")
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
