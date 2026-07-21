from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Cash index — reliable free 15m OHLC via Yahoo (max ~60 days).
# Futures continuous tickers (FCE=F / CAC=F) are unreliable/unavailable on Yahoo.
DEFAULT_YAHOO_TICKER = "^FCHI"
DEFAULT_INTERVAL = "15m"
DEFAULT_PERIOD = "60d"


def fetch_yahoo_ohlc(
    *,
    ticker: str = DEFAULT_YAHOO_TICKER,
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """
    Download CAC40 OHLC from Yahoo Finance and return a normalized DataFrame
    indexed by timezone-aware timestamps with columns open,high,low,close[,volume].
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required: pip install yfinance") from exc

    raw = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"No Yahoo data for {ticker} ({interval}/{period}). "
            "15m history is limited to the last ~60 days."
        )

    df = raw.copy()
    # yfinance may return MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    rename = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "close",
        "volume": "volume",
    }
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Yahoo response missing columns: {missing}")

    keep = required + (["volume"] if "volume" in df.columns else [])
    df = df[keep].dropna(subset=required)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Paris")
    df = df.sort_index()
    logger.info("Fetched %s bars from Yahoo %s (%s/%s)", len(df), ticker, interval, period)
    return df


def yahoo_source_meta() -> dict[str, Any]:
    return {
        "provider": "Yahoo Finance",
        "ticker": DEFAULT_YAHOO_TICKER,
        "interval": DEFAULT_INTERVAL,
        "period": DEFAULT_PERIOD,
        "note": (
            "Free CAC 40 cash index (^FCHI), ~60 days of 15m bars. "
            "For multi-year history buy a futures CSV (e.g. backtestmarket MX 15m) and upload it."
        ),
    }
