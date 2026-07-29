"""Production IG Lightstreamer client: live PRICE ticks + TRADE wake-ups.

Pin ``lightstreamer-client-lib==1.0.3``. Ledger writes must not run on LS
callback threads — enqueue via callbacks only.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from chatbot.trader.ig_stream_probe import (
    TickBar,
    _bucket_15m,
    _parse_mid,
    lightstreamer_password,
)

logger = logging.getLogger(__name__)

_PRICE_FIELDS = ("BID", "OFFER", "HIGH", "LOW", "CHANGE", "MARKET_STATE", "UPDATE_TIME")
_TRADE_FIELDS = ("CONFIRMS", "OPU", "WOU")

OnTick = Callable[[dict[str, Any]], None]
OnBarClosed = Callable[[TickBar, dict[str, Any]], None]
OnTradeWakeup = Callable[[dict[str, Any]], None]
OnStatus = Callable[[str], None]


@dataclass
class StreamQuote:
    bid: float = 0.0
    offer: float = 0.0
    mid: float = 0.0
    market_state: str = ""
    ts: str = ""
    epic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bid": self.bid,
            "offer": self.offer,
            "mid": self.mid,
            "market_state": self.market_state,
            "ts": self.ts,
            "epic": self.epic,
        }


@dataclass
class IgStreamService:
    """Long-lived Lightstreamer session with reconnect for one epic/account."""

    endpoint: str
    account_id: str
    cst: str
    xst: str
    epic: str
    on_tick: OnTick | None = None
    on_bar_closed: OnBarClosed | None = None
    on_trade_wakeup: OnTradeWakeup | None = None
    on_status: OnStatus | None = None
    on_need_relogin: Callable[[], tuple[str, str, str] | None] | None = None
    reconnect_base_seconds: float = 2.0
    reconnect_max_seconds: float = 60.0

    ticks_total: int = 0
    bars_closed_total: int = 0
    trade_events_total: int = 0
    reconnect_count: int = 0
    last_error: str = ""
    last_tick_at: str = ""
    last_bar_closed_at: str = ""
    last_trade_at: str = ""
    last_quote: StreamQuote = field(default_factory=StreamQuote)

    _status: str = field(default="DISCONNECTED", init=False, repr=False)
    _client: Any = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _open_bar: TickBar | None = field(default=None, init=False, repr=False)
    _connected_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @property
    def status(self) -> str:
        return self._status

    @property
    def connected(self) -> bool:
        return "CONNECTED" in (self._status or "")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"ig-stream-{self.account_id}-{self.epic}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 10.0) -> None:
        self._stop.set()
        self._disconnect_client()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def update_tokens(self, *, cst: str, xst: str, endpoint: str | None = None) -> None:
        with self._lock:
            self.cst = cst
            self.xst = xst
            if endpoint:
                self.endpoint = endpoint.rstrip("/")

    def _set_status(self, status: str) -> None:
        self._status = status or "DISCONNECTED"
        if "CONNECTED" in self._status:
            self._connected_event.set()
        else:
            self._connected_event.clear()
        if self.on_status:
            try:
                self.on_status(self._status)
            except Exception:
                logger.exception("on_status callback failed")

    def _disconnect_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            client.disconnect()
        except Exception:
            logger.debug("Lightstreamer disconnect failed", exc_info=True)

    def _run_loop(self) -> None:
        backoff = self.reconnect_base_seconds
        while not self._stop.is_set():
            try:
                if self._connect_and_subscribe():
                    backoff = self.reconnect_base_seconds
                    while not self._stop.is_set() and self.connected:
                        time.sleep(0.5)
                    if self._stop.is_set():
                        break
                    self.reconnect_count += 1
                    self.last_error = f"disconnected:{self._status}"
                else:
                    self.last_error = f"connect_failed:{self._status}"
                    self.reconnect_count += 1
            except Exception as exc:
                self.last_error = str(exc)
                self.reconnect_count += 1
                logger.exception("IgStreamService loop error")
            self._disconnect_client()
            self._set_status("DISCONNECTED")
            if self._stop.is_set():
                break
            # Refresh tokens before retry when available.
            if self.on_need_relogin:
                try:
                    refreshed = self.on_need_relogin()
                    if refreshed:
                        cst, xst, endpoint = refreshed
                        self.update_tokens(cst=cst, xst=xst, endpoint=endpoint or None)
                except Exception:
                    logger.exception("on_need_relogin failed")
            time.sleep(backoff)
            backoff = min(self.reconnect_max_seconds, backoff * 2.0)
        self._disconnect_client()
        self._set_status("DISCONNECTED")

    def _connect_and_subscribe(self) -> bool:
        from lightstreamer.client import (
            ClientListener,
            LightstreamerClient,
            Subscription,
            SubscriptionListener,
        )

        endpoint = self.endpoint.rstrip("/")
        self._connected_event.clear()
        client = LightstreamerClient(endpoint, None)
        client.connectionDetails.setUser(self.account_id)
        client.connectionDetails.setPassword(
            lightstreamer_password(self.cst, self.xst)
        )
        outer = self

        class _Status(ClientListener):
            def onStatusChange(self, status: str) -> None:  # noqa: N802
                outer._set_status(status)
                logger.info("Lightstreamer %s status=%s", outer.epic, status)

            def onServerError(self, code: int, message: str) -> None:  # noqa: N802
                outer.last_error = f"server:{code}:{message}"
                logger.error("Lightstreamer server error %s: %s", code, message)

        client.addListener(_Status())
        self._client = client
        client.connect()
        if not self._connected_event.wait(timeout=20.0):
            return False

        class _PriceListener(SubscriptionListener):
            def onItemUpdate(self, update) -> None:  # noqa: N802
                outer._handle_price_update(update)

        class _TradeListener(SubscriptionListener):
            def onItemUpdate(self, update) -> None:  # noqa: N802
                outer._handle_trade_update(update)

        price_sub = Subscription("MERGE", [f"MARKET:{self.epic}"], list(_PRICE_FIELDS))
        price_sub.setRequestedSnapshot("yes")
        price_sub.addListener(_PriceListener())
        client.subscribe(price_sub)

        trade_sub = Subscription(
            "DISTINCT", [f"TRADE:{self.account_id}"], list(_TRADE_FIELDS)
        )
        trade_sub.addListener(_TradeListener())
        client.subscribe(trade_sub)
        return True

    def _handle_price_update(self, update: Any) -> None:
        fields = {f: update.getValue(f) for f in _PRICE_FIELDS}
        now = datetime.now(timezone.utc)
        now_s = now.isoformat()
        try:
            bid = float(fields.get("BID") or 0)
            offer = float(fields.get("OFFER") or 0)
        except (TypeError, ValueError):
            bid, offer = 0.0, 0.0
        mid = _parse_mid(fields)
        if mid is None:
            return
        market_state = str(fields.get("MARKET_STATE") or "").strip()
        quote = StreamQuote(
            bid=bid,
            offer=offer,
            mid=float(mid),
            market_state=market_state,
            ts=now_s,
            epic=self.epic,
        )
        with self._lock:
            self.ticks_total += 1
            self.last_tick_at = now_s
            self.last_quote = quote
            bucket = _bucket_15m(now)
            closed: TickBar | None = None
            if self._open_bar is None:
                self._open_bar = TickBar(
                    bucket_start=bucket,
                    open=mid,
                    high=mid,
                    low=mid,
                    close=mid,
                    ticks=1,
                )
            elif self._open_bar.bucket_start != bucket:
                closed = self._open_bar
                self._open_bar = TickBar(
                    bucket_start=bucket,
                    open=mid,
                    high=mid,
                    low=mid,
                    close=mid,
                    ticks=1,
                )
            else:
                bar = self._open_bar
                bar.high = max(bar.high, mid)
                bar.low = min(bar.low, mid)
                bar.close = mid
                bar.ticks += 1
        if self.on_tick:
            try:
                self.on_tick(quote.to_dict())
            except Exception:
                logger.exception("on_tick failed")
        if closed is not None:
            with self._lock:
                self.bars_closed_total += 1
                self.last_bar_closed_at = now_s
            if self.on_bar_closed:
                try:
                    self.on_bar_closed(closed, quote.to_dict())
                except Exception:
                    logger.exception("on_bar_closed failed")

    def _handle_trade_update(self, update: Any) -> None:
        fields = {f: update.getValue(f) for f in _TRADE_FIELDS}
        if not any(fields.get(k) for k in _TRADE_FIELDS):
            return
        now_s = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.trade_events_total += 1
            self.last_trade_at = now_s
        if self.on_trade_wakeup:
            try:
                self.on_trade_wakeup(fields)
            except Exception:
                logger.exception("on_trade_wakeup failed")
