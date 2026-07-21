from __future__ import annotations

import logging
from typing import Any

import httpx
import pandas as pd

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import (
    MarketSnapshot,
    OrderType,
    Side,
    WorkingOrder,
)

logger = logging.getLogger(__name__)

_IG_HOSTS = {
    "DEMO": "https://demo-api.ig.com/gateway/deal",
    "LIVE": "https://api.ig.com/gateway/deal",
}


class IgConnector:
    """
    IG Markets REST connector.

    Maintains a local HedgeLedger mirror for hedge-mode semantics.
    When dry_run=True (default for safety), orders are ledger-only.
    Live HTTP order placement is used when dry_run=False and credentials exist.
    """

    def __init__(self, config: Cac40Config, *, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self.ledger = HedgeLedger(config=config, symbol=config.symbol)
        self._cst: str | None = None
        self._security: str | None = None
        self._client = httpx.Client(timeout=30.0)

    @property
    def base_url(self) -> str:
        return _IG_HOSTS.get(self.config.ig_acc_type.upper(), _IG_HOSTS["DEMO"])

    def close(self) -> None:
        self._client.close()

    def login(self) -> None:
        if not self.config.ig_api_key or not self.config.ig_username:
            logger.warning("IG credentials missing; connector stays in offline/dry mode")
            return
        resp = self._client.post(
            f"{self.base_url}/session",
            headers=self._headers(version="2"),
            json={
                "identifier": self.config.ig_username,
                "password": self.config.ig_password,
            },
        )
        resp.raise_for_status()
        self._cst = resp.headers.get("CST")
        self._security = resp.headers.get("X-SECURITY-TOKEN")
        logger.info("IG session opened (%s)", self.config.ig_acc_type)

    def _headers(self, *, version: str = "1") -> dict[str, str]:
        h = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "VERSION": version,
            "X-IG-API-KEY": self.config.ig_api_key,
        }
        if self._cst:
            h["CST"] = self._cst
        if self._security:
            h["X-SECURITY-TOKEN"] = self._security
        return h

    def get_ohlc(self, timeframe: str, lookback: int) -> pd.DataFrame:
        resolution = {
            "15m": "MINUTE_15",
            "1h": "HOUR",
            "1H": "HOUR",
            "1d": "DAY",
            "1D": "DAY",
            "D": "DAY",
        }.get(timeframe, "MINUTE_15")
        if not self._cst:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        url = f"{self.base_url}/prices/{self.config.epic}/{resolution}/{lookback}"
        resp = self._client.get(url, headers=self._headers(version="3"))
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        rows = []
        for p in prices:
            snap = p.get("snapshotTimeUTC") or p.get("snapshotTime")
            mid_o = _mid(p.get("openPrice"))
            mid_h = _mid(p.get("highPrice"))
            mid_l = _mid(p.get("lowPrice"))
            mid_c = _mid(p.get("closePrice"))
            if None in (mid_o, mid_h, mid_l, mid_c):
                continue
            rows.append(
                {
                    "ts": pd.Timestamp(snap),
                    "open": mid_o,
                    "high": mid_h,
                    "low": mid_l,
                    "close": mid_c,
                    "volume": p.get("lastTradedVolume") or 0,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).set_index("ts").sort_index()
        return df

    def sync_price(self) -> float:
        df = self.get_ohlc("15m", 2)
        if df.empty:
            return self.ledger.last_price
        price = float(df["close"].iloc[-1])
        self.ledger.last_price = price
        self.ledger.mark_to_market(price)
        return price

    def get_snapshot(self) -> MarketSnapshot:
        if self.ledger.last_price <= 0:
            self.sync_price()
        return self.ledger.get_snapshot()

    def place_order(self, order: WorkingOrder) -> WorkingOrder:
        placed = self.ledger.place_order(order)
        if self.dry_run or not self._cst:
            logger.info("IG dry-run place %s", placed.to_dict())
            return placed
        body = self._ig_working_order_body(placed)
        resp = self._client.post(
            f"{self.base_url}/workingorders/otc",
            headers=self._headers(version="2"),
            json=body,
        )
        resp.raise_for_status()
        deal = resp.json()
        placed.client_ref = str(deal.get("dealReference") or placed.client_ref)
        return placed

    def amend_order(self, order_id: str, *, level: float) -> WorkingOrder:
        order = self.ledger.amend_order(order_id, level=level)
        if self.dry_run or not self._cst:
            return order
        # IG amend requires dealId; keep ledger as source in V1 dry path
        logger.info("IG amend requested for %s -> %s", order_id, level)
        return order

    def cancel_order(self, order_id: str) -> None:
        self.ledger.cancel_order(order_id)
        if self.dry_run or not self._cst:
            return
        logger.info("IG cancel requested for %s", order_id)

    def close_position(
        self,
        position_id: str,
        *,
        order_type: OrderType = OrderType.LIMIT,
        level: float | None = None,
    ) -> None:
        leg = self.ledger.positions.get(position_id)
        if not leg:
            return
        if order_type == OrderType.MARKET or level is None:
            self.ledger.market_close(position_id)
            return
        close_side = Side.SELL if leg.side == Side.BUY else Side.BUY
        from chatbot.cac40.models import OrderPurpose

        self.place_order(
            WorkingOrder(
                id="",
                type=OrderType.LIMIT,
                side=close_side,
                level=level,
                size=leg.size,
                purpose=OrderPurpose.TP,
                position_id=position_id,
            )
        )

    def market_open(self, side: Side, size: float) -> str:
        return self.ledger.market_open(side, size)

    def market_close(self, position_id: str) -> None:
        self.ledger.market_close(position_id)

    def _ig_working_order_body(self, order: WorkingOrder) -> dict[str, Any]:
        return {
            "epic": self.config.epic,
            "expiry": "-",
            "direction": "BUY" if order.side == Side.BUY else "SELL",
            "size": order.size,
            "level": order.level,
            "type": "LIMIT" if order.type == OrderType.LIMIT else "STOP",
            "currencyCode": "EUR",
            "timeInForce": "GOOD_TILL_CANCELLED",
            "guaranteedStop": False,
            "forceOpen": True,
        }


def _mid(price_obj: dict | None) -> float | None:
    if not price_obj:
        return None
    bid, ask = price_obj.get("bid"), price_obj.get("ask")
    if bid is None and ask is None:
        return None
    if bid is None:
        return float(ask)
    if ask is None:
        return float(bid)
    return (float(bid) + float(ask)) / 2.0
