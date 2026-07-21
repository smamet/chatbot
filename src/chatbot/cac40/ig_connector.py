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


class IgApiError(RuntimeError):
    """IG HTTP failure with a human-readable message."""


# Back-compat alias used by connector tests / callers.
IgAuthError = IgApiError


def format_ig_http_error(resp: httpx.Response, *, action: str, url: str) -> str:
    """Build a detailed error string from an IG HTTP response."""
    status = resp.status_code
    body_text = (resp.text or "").strip()
    error_code = ""
    detail = body_text
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            error_code = str(
                payload.get("errorCode")
                or payload.get("error")
                or payload.get("message")
                or ""
            ).strip()
            detail = error_code or str(payload)
    except Exception:
        if body_text.lstrip().lower().startswith("<!doctype") or "<html" in body_text[:80].lower():
            detail = "(HTML 404 page — usually wrong endpoint path or API version)"

    hints: list[str] = []
    code_l = error_code.lower()
    if status == 401 or "invalid-details" in code_l or "authentication" in code_l:
        hints.append("Check username/password (Web API demo login details, not email).")
        hints.append("API key must be created for the same environment (Demo vs Live).")
        hints.append("Demo key + Demo env, or Live key + Live env — do not mix.")
    elif status == 403:
        hints.append("API key may be disabled, or this account cannot use the REST API.")
    elif status == 404:
        hints.append("Confirm epic exists on Demo (search the market in IG) and Environment=Demo.")
        hints.append("Prices use GET /prices/{epic}?resolution=&max= (API v3).")

    lines = [
        f"IG {action} failed: HTTP {status}",
        f"URL: {url}",
    ]
    if error_code:
        lines.append(f"IG errorCode: {error_code}")
    elif detail and detail != error_code:
        lines.append(f"Body: {detail[:300]}")
    for hint in hints:
        lines.append(f"Hint: {hint}")
    return "\n".join(lines)


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
        url = f"{self.base_url}/session"
        resp = self._client.post(
            url,
            headers=self._headers(version="2"),
            json={
                "identifier": self.config.ig_username,
                "password": self.config.ig_password,
            },
        )
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="login", url=url))
        self._cst = resp.headers.get("CST")
        self._security = resp.headers.get("X-SECURITY-TOKEN")
        logger.info("IG session opened (%s)", self.config.ig_acc_type)
        if self.config.ig_account_id:
            self.switch_account(self.config.ig_account_id)

    def switch_account(self, account_id: str, *, default: bool = False) -> None:
        """Switch the active IG account for this session (needed for demo CFD accounts)."""
        account_id = (account_id or "").strip()
        if not account_id or not self._cst:
            return
        url = f"{self.base_url}/session"
        resp = self._client.put(
            url,
            headers=self._headers(version="1"),
            json={"accountId": account_id, "defaultAccount": default},
        )
        if resp.is_error:
            # Already on this account after login — not a failure.
            try:
                code = str((resp.json() or {}).get("errorCode") or "")
            except Exception:
                code = ""
            if resp.status_code == 412 and "accountId-must-be-different" in code:
                logger.info("IG already on account %s", account_id)
                return
            raise IgApiError(format_ig_http_error(resp, action="switch_account", url=url))
        if resp.headers.get("CST"):
            self._cst = resp.headers.get("CST")
        if resp.headers.get("X-SECURITY-TOKEN"):
            self._security = resp.headers.get("X-SECURITY-TOKEN")
        logger.info("IG switched to account %s", account_id)

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
        # IG API v3: GET /prices/{epic}?resolution=&max=&pageSize=
        # (path /prices/{epic}/{resolution}/{n} is v2 only — v3 returns HTML 404)
        n = max(1, int(lookback))
        url = f"{self.base_url}/prices/{self.config.epic}"
        resp = self._client.get(
            url,
            headers=self._headers(version="3"),
            params={"resolution": resolution, "max": n, "pageSize": n},
        )
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="prices", url=str(resp.request.url)))
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
