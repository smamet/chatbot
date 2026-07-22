from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.ig_connector import IgConnector

logger = logging.getLogger(__name__)


def ig_config_from_connector(config: dict[str, Any] | None) -> Cac40Config:
    """Map saved IG connector config keys onto Cac40Config."""
    cfg = dict(config or {})
    return Cac40Config(
        ig_api_key=str(cfg.get("api_key") or ""),
        ig_username=str(cfg.get("username") or ""),
        ig_password=str(cfg.get("password") or ""),
        ig_account_id=str(cfg.get("account_id") or ""),
        ig_acc_type=str(cfg.get("acc_type") or "DEMO").upper() or "DEMO",
        epic=str(cfg.get("epic") or Cac40Config().epic),
    )


def _gap_bar_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    if a.tzinfo is not None and b.tzinfo is not None and a.tzinfo != b.tzinfo:
        b = b.tz_convert(a.tzinfo)
    if b <= a:
        return 1
    return max(1, int((b - a) / pd.Timedelta(minutes=15)) + 1)


def catchup_ohlc_15m(
    connector: IgConnector,
    *,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    timezone: str = "Europe/Paris",
) -> tuple[pd.DataFrame, str]:
    """
    Fetch 15m bars to catch up a local CSV.

    1. Paged ``from``/``to`` range (start inclusive — callers drop ``<= last``).
    2. If empty: fallback ``max=N`` recent bars (DEMO often serves ``max`` when
       date-range returns nothing; also uses fewer points for small gaps).

    Returns ``(dataframe_in_timezone, mode)`` where mode is
    ``range`` | ``max_fallback`` | ``empty``.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    # Prefer last *closed* 15m as range end — open candle / "now" confuses IG.
    end_local = end_ts.tz_convert(timezone)
    floored = end_local.floor("15min")
    range_end = floored - pd.Timedelta(minutes=15)
    if range_end.tzinfo is None:
        range_end = range_end.tz_localize(timezone)

    range_to = range_end if range_end > pd.Timestamp(start_ts) else end_ts
    df = connector.fetch_ohlc_range(
        timeframe="15m",
        start=start_ts,
        end=range_to,
        timezone=timezone,
    )
    start_cmp = pd.Timestamp(start_ts)
    if start_cmp.tzinfo is None:
        start_cmp = start_cmp.tz_localize(timezone)
    else:
        start_cmp = start_cmp.tz_convert(timezone)

    def _has_newer(frame: pd.DataFrame | None) -> bool:
        if frame is None or frame.empty:
            return False
        idx = frame.index
        if idx.tz is None:
            return bool((idx.tz_localize(timezone) > start_cmp).any())
        return bool((idx.tz_convert(timezone) > start_cmp).any())

    if _has_newer(df):
        return df, "range"

    n = min(500, max(64, _gap_bar_count(start_ts, range_end) + 16))
    logger.info(
        "IG range empty/no-newer (%s → %s); falling back to max=%s",
        start_ts,
        range_end,
        n,
    )
    tip = connector.get_ohlc("15m", n)
    if tip is None or tip.empty:
        empty = tip if tip is not None else (df if df is not None else pd.DataFrame())
        return empty, "empty"
    if tip.index.tz is None:
        tip.index = tip.index.tz_localize("UTC")
    tip = tip.copy()
    tip.index = tip.index.tz_convert(timezone)
    if not _has_newer(tip):
        return tip, "empty"
    return tip, "max_fallback"


def fetch_ig_ohlc_range(
    ig_config: dict[str, Any],
    *,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    timeframe: str = "15m",
    timezone: str = "Europe/Paris",
    allowance_out: dict[str, Any] | None = None,
    allow_max_fallback: bool = True,
) -> pd.DataFrame:
    """Login to IG and fetch mid-price OHLC for [start, end] (with max= fallback)."""
    config = ig_config_from_connector(ig_config)
    if not config.ig_api_key or not config.ig_username or not config.ig_password:
        raise ValueError("IG connector is missing api_key, username, or password")
    connector = IgConnector(config, dry_run=True)
    try:
        connector.login()
        if not connector.authenticated:
            raise ValueError("IG login did not establish a session")
        if timeframe == "15m" and allow_max_fallback:
            df, mode = catchup_ohlc_15m(
                connector, start=start, end=end, timezone=timezone
            )
            logger.info("IG OHLC catch-up mode=%s bars=%s", mode, 0 if df is None else len(df))
        else:
            df = connector.fetch_ohlc_range(
                timeframe=timeframe,
                start=start,
                end=end,
                timezone=timezone,
            )
        if allowance_out is not None and connector.last_price_allowance:
            allowance_out.clear()
            allowance_out.update(connector.last_price_allowance)
            remaining = allowance_out.get("remaining") or allowance_out.get(
                "remainingAllowance"
            )
            expiry = allowance_out.get("expiry") or allowance_out.get("allowanceExpiry")
            logger.info(
                "IG historical allowance remaining=%s expiry=%s", remaining, expiry
            )
        return df if df is not None else pd.DataFrame()
    finally:
        connector.close()
