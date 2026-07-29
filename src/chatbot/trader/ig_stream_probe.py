"""DEMO IG Lightstreamer probe: live prices, tick→15m HLC, TRADE confirms.

Orders are still placed/cancelled via REST; this module validates streaming.
Pin ``lightstreamer-client-lib==1.0.3`` (IG LS server compatibility).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_EPICS = ("IX.D.CAC.BMU.IP", "CS.D.EURUSD.MINI.IP")
_PRICE_FIELDS = ("BID", "OFFER", "HIGH", "LOW", "CHANGE", "MARKET_STATE", "UPDATE_TIME")
_TRADE_FIELDS = ("CONFIRMS", "OPU", "WOU")


@dataclass
class TickBar:
    """In-memory 15m HLC assembled from streamed bid/offer ticks."""

    bucket_start: datetime
    open: float
    high: float
    low: float
    close: float
    ticks: int = 0


@dataclass
class StreamProbeResult:
    ok: bool
    message: str
    ticks_by_epic: dict[str, int] = field(default_factory=dict)
    bars_by_epic: dict[str, int] = field(default_factory=dict)
    trade_updates: int = 0
    lightstreamer_endpoint: str = ""
    account_id: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "ticks_by_epic": dict(self.ticks_by_epic),
            "bars_by_epic": dict(self.bars_by_epic),
            "trade_updates": self.trade_updates,
            "lightstreamer_endpoint": self.lightstreamer_endpoint,
            "account_id": self.account_id,
            "error": self.error,
        }


def lightstreamer_password(cst: str, xst: str) -> str:
    return f"CST-{cst}|XST-{xst}"


def _bucket_15m(ts: datetime) -> datetime:
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


def _parse_mid(fields: dict[str, Any]) -> float | None:
    try:
        bid = float(fields.get("BID") or 0)
        offer = float(fields.get("OFFER") or 0)
    except (TypeError, ValueError):
        return None
    if bid > 0 and offer > 0:
        return (bid + offer) / 2.0
    if bid > 0:
        return bid
    if offer > 0:
        return offer
    return None


class _TradeCollector:
    """Thread-safe collector for TRADE subscription updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.updates: list[dict[str, Any]] = []

    def on_update(self, fields: dict[str, Any]) -> None:
        with self._lock:
            self.updates.append(dict(fields))

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.updates)

    def wait_for(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            with self._lock:
                for row in self.updates:
                    if predicate(row):
                        return row
            time.sleep(0.05)
        return None


class IgLightstreamerSession:
    """Thin wrapper around official LightstreamerClient for IG DEMO probes."""

    def __init__(
        self,
        *,
        endpoint: str,
        account_id: str,
        cst: str,
        xst: str,
    ) -> None:
        from lightstreamer.client import ClientListener, LightstreamerClient, Subscription, SubscriptionListener

        self._ClientListener = ClientListener
        self._Subscription = Subscription
        self._SubscriptionListener = SubscriptionListener
        self.endpoint = endpoint.rstrip("/")
        self.account_id = account_id
        self.trade = _TradeCollector()
        self.ticks_by_epic: dict[str, int] = defaultdict(int)
        self._bars: dict[str, dict[datetime, TickBar]] = defaultdict(dict)
        self._status = "DISCONNECTED"
        self._status_event = threading.Event()
        self._client = LightstreamerClient(self.endpoint, None)
        self._client.connectionDetails.setUser(account_id)
        self._client.connectionDetails.setPassword(lightstreamer_password(cst, xst))

        outer = self

        class _Status(ClientListener):
            def onStatusChange(self, status: str) -> None:  # noqa: N802
                outer._status = status
                if "CONNECTED" in (status or ""):
                    outer._status_event.set()
                logger.info("Lightstreamer status: %s", status)

            def onServerError(self, code: int, message: str) -> None:  # noqa: N802
                logger.error("Lightstreamer server error %s: %s", code, message)

        self._client.addListener(_Status())

    @property
    def status(self) -> str:
        return self._status

    def connect(self, *, timeout: float = 20.0) -> bool:
        self._status_event.clear()
        self._client.connect()
        return self._status_event.wait(timeout=timeout)

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
        except Exception:
            logger.debug("Lightstreamer disconnect failed", exc_info=True)

    def subscribe_prices(self, epics: list[str]) -> None:
        items = [f"MARKET:{e}" for e in epics if e]
        if not items:
            return
        outer = self

        class _PriceListener(self._SubscriptionListener):
            def onItemUpdate(self, update) -> None:  # noqa: N802
                item = str(update.getItemName() or "")
                epic = item.split(":", 1)[-1] if ":" in item else item
                fields = {f: update.getValue(f) for f in _PRICE_FIELDS}
                outer.ticks_by_epic[epic] += 1
                mid = _parse_mid(fields)
                if mid is None:
                    return
                now = datetime.now(timezone.utc)
                bucket = _bucket_15m(now)
                bars = outer._bars[epic]
                bar = bars.get(bucket)
                if bar is None:
                    bars[bucket] = TickBar(
                        bucket_start=bucket,
                        open=mid,
                        high=mid,
                        low=mid,
                        close=mid,
                        ticks=1,
                    )
                else:
                    bar.high = max(bar.high, mid)
                    bar.low = min(bar.low, mid)
                    bar.close = mid
                    bar.ticks += 1

        sub = self._Subscription("MERGE", items, list(_PRICE_FIELDS))
        sub.setRequestedSnapshot("yes")
        sub.addListener(_PriceListener())
        self._client.subscribe(sub)

    def subscribe_trade(self) -> None:
        outer = self

        class _TradeListener(self._SubscriptionListener):
            def onItemUpdate(self, update) -> None:  # noqa: N802
                fields = {f: update.getValue(f) for f in _TRADE_FIELDS}
                outer.trade.on_update(fields)

        sub = self._Subscription(
            "DISTINCT",
            [f"TRADE:{self.account_id}"],
            list(_TRADE_FIELDS),
        )
        sub.addListener(_TradeListener())
        self._client.subscribe(sub)

    def bar_counts(self) -> dict[str, int]:
        return {epic: len(bars) for epic, bars in self._bars.items()}


def run_ig_stream_probe(
    config: dict[str, Any],
    *,
    epics: list[str] | None = None,
    seconds: float = 45.0,
) -> StreamProbeResult:
    """Connect Lightstreamer, subscribe PRICE + TRADE, assemble 15m HLC from ticks."""
    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgConnector

    api_key = str(config.get("api_key", "")).strip()
    username = str(config.get("username", "")).strip()
    password = str(config.get("password", "")).strip()
    if not api_key or not username or not password:
        return StreamProbeResult(
            ok=False,
            message="IG API key, username, and password are required.",
            error="missing_credentials",
        )
    acc_type = str(config.get("acc_type", "DEMO") or "DEMO").strip().upper()
    if acc_type != "DEMO":
        return StreamProbeResult(
            ok=False,
            message="Stream probe is DEMO-only.",
            error="live_blocked",
        )
    epic_list = [e.strip() for e in (epics or list(_DEFAULT_EPICS)) if e and e.strip()]
    if not epic_list:
        epic_list = list(_DEFAULT_EPICS)
    account_id = str(config.get("account_id", "")).strip()
    cfg = TraderConfig(
        ig_api_key=api_key,
        ig_username=username,
        ig_password=password,
        ig_account_id=account_id,
        ig_acc_type=acc_type,
        epic=epic_list[0],
    )
    ig = IgConnector(cfg, dry_run=False)
    lines: list[str] = []
    try:
        ig.login()
        if not ig._cst or not ig._security:
            return StreamProbeResult(
                ok=False, message="IG login failed (no session tokens).", error="no_session"
            )
        endpoint = ig.lightstreamer_endpoint or ""
        acct = (ig.current_account_id or account_id or "").strip()
        if not endpoint:
            return StreamProbeResult(
                ok=False,
                message="Login OK but lightstreamerEndpoint missing from /session.",
                error="no_ls_endpoint",
            )
        if not acct:
            return StreamProbeResult(
                ok=False,
                message="No account id for Lightstreamer (set account_id or use session default).",
                error="no_account",
            )
        session = IgLightstreamerSession(
            endpoint=endpoint,
            account_id=acct,
            cst=ig._cst,
            xst=ig._security,
        )
        connected = session.connect(timeout=20.0)
        lines.append(f"Lightstreamer endpoint={endpoint}")
        lines.append(f"account={acct} · connected={connected} · status={session.status}")
        if not connected:
            session.disconnect()
            return StreamProbeResult(
                ok=False,
                message="\n".join(lines + ["Failed to reach CONNECTED:* status."]),
                lightstreamer_endpoint=endpoint,
                account_id=acct,
                error="ls_connect_failed",
            )
        session.subscribe_prices(epic_list)
        session.subscribe_trade()
        time.sleep(max(5.0, float(seconds)))
        ticks = dict(session.ticks_by_epic)
        bars = session.bar_counts()
        trade_n = session.trade.count
        session.disconnect()
        lines.append(f"epics={','.join(epic_list)}")
        lines.append(f"ticks={json.dumps(ticks)}")
        lines.append(f"15m_bar_buckets={json.dumps(bars)}")
        lines.append(f"trade_updates={trade_n}")
        ok = sum(ticks.values()) > 0
        if not ok:
            lines.append("No PRICE ticks received (market closed or item schema mismatch).")
        return StreamProbeResult(
            ok=ok,
            message="\n".join(lines),
            ticks_by_epic=ticks,
            bars_by_epic=bars,
            trade_updates=trade_n,
            lightstreamer_endpoint=endpoint,
            account_id=acct,
            error=None if ok else "no_ticks",
        )
    except Exception as exc:
        return StreamProbeResult(
            ok=False,
            message=f"Stream probe failed: {exc}",
            error="exception",
        )
    finally:
        ig.close()
