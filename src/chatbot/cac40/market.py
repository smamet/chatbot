from __future__ import annotations

from typing import Protocol

import pandas as pd

from chatbot.cac40.models import MarketSnapshot, OrderType, Side, WorkingOrder


class MarketConnector(Protocol):
    def get_ohlc(self, timeframe: str, lookback: int) -> pd.DataFrame:
        """Return OHLC indexed by timestamp with columns open,high,low,close[,volume]."""

    def get_snapshot(self) -> MarketSnapshot: ...

    def place_order(self, order: WorkingOrder) -> WorkingOrder: ...

    def amend_order(
        self, order_id: str, *, level: float, size: float | None = None
    ) -> WorkingOrder: ...

    def cancel_order(self, order_id: str) -> None: ...

    def close_position(
        self,
        position_id: str,
        *,
        order_type: OrderType = OrderType.LIMIT,
        level: float | None = None,
    ) -> None: ...

    def market_open(self, side: Side, size: float) -> str:
        """Open a new leg at market. Returns position id."""

    def market_close(self, position_id: str) -> None: ...
