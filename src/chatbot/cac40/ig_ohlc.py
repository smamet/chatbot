from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.ig_connector import IgConnector


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


def fetch_ig_ohlc_range(
    ig_config: dict[str, Any],
    *,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    timeframe: str = "15m",
    timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    """Login to IG and fetch mid-price OHLC for [start, end]."""
    config = ig_config_from_connector(ig_config)
    if not config.ig_api_key or not config.ig_username or not config.ig_password:
        raise ValueError("IG connector is missing api_key, username, or password")
    connector = IgConnector(config, dry_run=True)
    try:
        connector.login()
        if not connector.authenticated:
            raise ValueError("IG login did not establish a session")
        return connector.fetch_ohlc_range(
            timeframe=timeframe,
            start=start,
            end=end,
            timezone=timezone,
        )
    finally:
        connector.close()
