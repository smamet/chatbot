from __future__ import annotations

import logging
import time
from datetime import datetime
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

_EMPTY_OHLC = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

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

    @property
    def authenticated(self) -> bool:
        return bool(self._cst)

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
        resolution = _resolution_for_timeframe(timeframe)
        if not self._cst:
            return _EMPTY_OHLC.copy()
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
        return _prices_payload_to_df(resp.json().get("prices") or [])

    def fetch_ohlc_range(
        self,
        *,
        timeframe: str = "15m",
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        page_size: int = 500,
        page_wait_seconds: float = 0.25,
        timezone: str = "Europe/Paris",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC between start/end via IG API v3 (paged).

        Dates are sent as UTC ``YYYY-MM-DDTHH:MM:SS``. Large windows are split
        into ~7-day chunks to stay within IG allowance limits.
        """
        resolution = _resolution_for_timeframe(timeframe)
        if not self._cst:
            return _EMPTY_OHLC.copy()

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        if end_ts <= start_ts:
            return _EMPTY_OHLC.copy()

        frames: list[pd.DataFrame] = []
        chunk_start = start_ts
        chunk_delta = pd.Timedelta(days=7)
        while chunk_start < end_ts:
            chunk_end = min(chunk_start + chunk_delta, end_ts)
            frames.append(
                self._fetch_ohlc_window(
                    resolution=resolution,
                    start=chunk_start,
                    end=chunk_end,
                    page_size=page_size,
                    page_wait_seconds=page_wait_seconds,
                )
            )
            chunk_start = chunk_end

        if not frames:
            return _EMPTY_OHLC.copy()
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(timezone)
        return df

    def _fetch_ohlc_window(
        self,
        *,
        resolution: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        page_size: int,
        page_wait_seconds: float,
    ) -> pd.DataFrame:
        url = f"{self.base_url}/prices/{self.config.epic}"
        page_number = 1
        all_prices: list[dict[str, Any]] = []
        while True:
            params = {
                "resolution": resolution,
                "from": _ig_api_ts(start),
                "to": _ig_api_ts(end),
                "pageSize": max(1, int(page_size)),
                "pageNumber": page_number,
            }
            resp = self._client.get(
                url,
                headers=self._headers(version="3"),
                params=params,
            )
            if resp.is_error:
                raise IgApiError(
                    format_ig_http_error(resp, action="prices", url=str(resp.request.url))
                )
            payload = resp.json() or {}
            prices = payload.get("prices") or []
            all_prices.extend(prices)
            page_data = (payload.get("metadata") or {}).get("pageData") or {}
            total_pages = int(page_data.get("totalPages") or 1)
            current = int(page_data.get("pageNumber") or page_number)
            if current >= total_pages or not prices:
                break
            page_number = current + 1
            if page_wait_seconds > 0:
                time.sleep(page_wait_seconds)
        return _prices_payload_to_df(all_prices)

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


def _resolution_for_timeframe(timeframe: str) -> str:
    return {
        "15m": "MINUTE_15",
        "1h": "HOUR",
        "1H": "HOUR",
        "1d": "DAY",
        "1D": "DAY",
        "D": "DAY",
    }.get(timeframe, "MINUTE_15")


def _ig_api_ts(ts: pd.Timestamp) -> str:
    utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return utc.strftime("%Y-%m-%dT%H:%M:%S")


def _prices_payload_to_df(prices: list[dict[str, Any]]) -> pd.DataFrame:
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
        return _EMPTY_OHLC.copy()
    return pd.DataFrame(rows).set_index("ts").sort_index()


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
