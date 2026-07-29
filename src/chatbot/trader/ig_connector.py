from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import httpx
import pandas as pd

from chatbot.trader.config import TraderConfig
from chatbot.trader.hedge_ledger import HedgeLedger
from chatbot.trader.models import (
    LegRole,
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

# Live STOP/LIMIT rejects used ATTACHED_ORDER_LEVEL_ERROR both when too close and
# when DEMO probes sat ~84pts away (possible maxStopOrLimitDistance). Clearance
# clamps into [min, max] from dealingRules; floors are only lower bounds.
_DEFAULT_STOP_CLEARANCE_POINTS = 12.0
_DEFAULT_LIMIT_CLEARANCE_POINTS = 12.0
_DEFAULT_MAX_CLEARANCE_POINTS = 100.0


class IgApiError(RuntimeError):
    """IG HTTP failure with a human-readable message."""


# Back-compat alias used by connector tests / callers.
IgAuthError = IgApiError


def compact_ig_error(exc: BaseException) -> dict[str, Any]:
    """Parse an IG exception into a short structured dict for mirror/UI logs."""
    msg = str(exc).strip()
    error_code = ""
    http_status: int | None = None
    for line in msg.splitlines():
        if "errorCode:" in line:
            error_code = line.split("errorCode:", 1)[-1].strip().split()[0]
        if line.startswith("IG ") and " failed: HTTP " in line:
            try:
                http_status = int(line.rsplit("HTTP ", 1)[-1].strip().split()[0])
            except (TypeError, ValueError):
                pass
    head = msg.split("\n", 1)[0]
    if len(head) > 160:
        head = head[:159] + "…"
    out: dict[str, Any] = {"error": head or msg[:160]}
    if error_code:
        out["error_code"] = error_code
    if http_status is not None:
        out["http_status"] = http_status
    return out


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
        if "historical-data-allowance" in code_l:
            hints.append(
                "Weekly IG historical price-point allowance is exhausted for this account "
                "(DEMO ~10k/week). Wait for reset, or reuse the local OHLC CSV."
            )
        else:
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

    def __init__(self, config: TraderConfig, *, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self.ledger = HedgeLedger(config=config, symbol=config.symbol)
        self._cst: str | None = None
        self._security: str | None = None
        self._client = httpx.Client(timeout=30.0)
        # From /session JSON — required for Lightstreamer (never hard-code the host).
        self.lightstreamer_endpoint: str | None = None
        self.current_account_id: str | None = None
        # Last /prices metadata.allowance (remainingAllowance, totalAllowance, …).
        self.last_price_allowance: dict[str, Any] | None = None
        # Compact last deal/confirm snapshot for mirror / cycle ops log (not raw HTTP).
        self.last_ig_result: dict[str, Any] = {}
        # Dealable quote from already-fetched positions/markets (no extra HTTP).
        self.last_dealable_bid: float | None = None
        self.last_dealable_offer: float | None = None
        self._market_cache: dict[str, dict[str, Any]] = {}

    def _note_ig_result(self, **fields: Any) -> dict[str, Any]:
        self.last_ig_result = {
            k: v for k, v in fields.items() if v is not None and v != ""
        }
        return self.last_ig_result

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
        try:
            body = resp.json() if resp.content else {}
        except Exception:
            body = {}
        if isinstance(body, dict):
            ls = str(body.get("lightstreamerEndpoint") or "").strip()
            self.lightstreamer_endpoint = ls or None
            acct = str(body.get("currentAccountId") or "").strip()
            if not acct:
                accounts = body.get("accounts") or []
                if isinstance(accounts, list) and accounts:
                    first = accounts[0] if isinstance(accounts[0], dict) else {}
                    acct = str(first.get("accountId") or "").strip()
            self.current_account_id = acct or None
        logger.info(
            "IG session opened (%s) ls=%s account=%s",
            self.config.ig_acc_type,
            self.lightstreamer_endpoint or "—",
            self.current_account_id or self.config.ig_account_id or "—",
        )
        if self.config.ig_account_id:
            self.switch_account(self.config.ig_account_id)
            self.current_account_id = self.config.ig_account_id.strip() or self.current_account_id

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
        payload = resp.json() or {}
        self.last_price_allowance = _extract_price_allowance(payload)
        return _prices_payload_to_df(payload.get("prices") or [])

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
            allowance = _extract_price_allowance(payload)
            if allowance:
                self.last_price_allowance = allowance
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

    def get_accounts(self) -> list[dict[str, Any]]:
        if not self._cst:
            return []
        url = f"{self.base_url}/accounts"
        resp = self._client.get(url, headers=self._headers(version="1"))
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="accounts", url=url))
        payload = resp.json() if resp.content else {}
        rows = (payload or {}).get("accounts") or []
        return [r for r in rows if isinstance(r, dict)]

    def get_active_account(self) -> dict[str, Any]:
        """Return the account matching ig_account_id, else preferred/enabled."""
        accounts = self.get_accounts()
        wanted = str(self.config.ig_account_id or "").strip()
        if wanted:
            for acc in accounts:
                if str(acc.get("accountId") or "").strip() == wanted:
                    return acc
        for acc in accounts:
            if acc.get("preferred"):
                return acc
        for acc in accounts:
            if str(acc.get("accountStatus") or "").upper() == "ENABLED":
                return acc
        return accounts[0] if accounts else {}

    def resolve_account_type(self) -> str:
        """CFD | SPREADBET | PHYSICAL | … from GET /accounts."""
        acc = self.get_active_account()
        return str(acc.get("accountType") or "").strip().upper()

    def get_market(self, epic: str | None = None) -> dict[str, Any]:
        """GET /markets/{epic} — instrument currencies, min size, etc."""
        ep = (epic or self.config.epic or "").strip()
        if not ep:
            return {}
        if ep in self._market_cache:
            self._remember_dealable_from_market(self._market_cache[ep])
            return self._market_cache[ep]
        if not self._cst:
            return {}
        url = f"{self.base_url}/markets/{ep}"
        resp = self._client.get(url, headers=self._headers(version="3"))
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="markets", url=url))
        payload = resp.json() if resp.content else {}
        market = payload if isinstance(payload, dict) else {}
        self._market_cache[ep] = market
        self._remember_dealable_from_market(market)
        return market

    def _remember_dealable_from_market(self, market: dict[str, Any]) -> None:
        snap = market.get("snapshot") if isinstance(market, dict) else None
        if not isinstance(snap, dict):
            return
        bid = _float_or_none(snap.get("bid"))
        offer = _float_or_none(snap.get("offer") if snap.get("offer") is not None else snap.get("ask"))
        if bid is not None and bid > 0:
            self.last_dealable_bid = bid
        if offer is not None and offer > 0:
            self.last_dealable_offer = offer

    def dealable_quote(self) -> tuple[float, float]:
        """Bid/offer from cached fetches, else last_price ± half spread."""
        bid = self.last_dealable_bid
        offer = self.last_dealable_offer
        if bid is not None and offer is not None and bid > 0 and offer > 0:
            return float(bid), float(offer)
        mid = float(self.ledger.last_price or 0)
        half = max(0.0, float(self.config.spread_points or 0) / 2.0)
        if mid <= 0:
            return 0.0, 0.0
        return mid - half, mid + half

    def resolve_point_size(self, *, epic: str | None = None) -> float:
        """Price value of one IG dealingRules POINTS unit.

        Indices: 1 point ≈ 1.0 in price. FX majors (EURUSD…): typically
        ``1 / scalingFactor`` (often 0.0001). Without scalingFactor, infer from mid.
        """
        market = self.get_market(epic) if self._cst else {}
        if not market:
            ep = (epic or self.config.epic or "").strip()
            market = self._market_cache.get(ep) or {}
        snapshot = (market.get("snapshot") or {}) if isinstance(market, dict) else {}
        raw_sf = snapshot.get("scalingFactor")
        try:
            sf = float(raw_sf) if raw_sf is not None else 0.0
        except (TypeError, ValueError):
            sf = 0.0
        if sf > 1.0:
            return 1.0 / sf
        mid = float(self.ledger.last_price or 0.0)
        if mid <= 0:
            try:
                bid = float(snapshot.get("bid") or 0)
                offer = float(snapshot.get("offer") or 0)
                if bid > 0 and offer > 0:
                    mid = (bid + offer) / 2.0
            except (TypeError, ValueError):
                mid = 0.0
        from chatbot.trader.point_size import infer_point_size

        return infer_point_size(mid)

    def resolve_min_stop_or_limit_distance(self, *, epic: str | None = None) -> float:
        """minNormalStopOrLimitDistance converted to **price** units (0 if unknown)."""
        return self._dealing_distance_points(
            "minNormalStopOrLimitDistance", epic=epic
        )

    def resolve_max_stop_or_limit_distance(self, *, epic: str | None = None) -> float:
        """maxStopOrLimitDistance converted to **price** units (0 if unknown)."""
        return self._dealing_distance_points("maxStopOrLimitDistance", epic=epic)

    def _dealing_distance_points(self, key: str, *, epic: str | None = None) -> float:
        market = self.get_market(epic) if self._cst else {}
        if not market:
            ep = (epic or self.config.epic or "").strip()
            market = self._market_cache.get(ep) or {}
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        row = rules.get(key) or {}
        if not isinstance(row, dict) or row.get("value") is None:
            return 0.0
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            return 0.0
        if value <= 0:
            return 0.0
        unit = str(row.get("unit") or "POINTS").strip().upper()
        if unit == "PERCENTAGE":
            mid = float(self.ledger.last_price or 0)
            return (mid * value / 100.0) if mid > 0 else 0.0
        # POINTS → price (FX: 2 pts @ 0.0001 = 0.0002; indices: 2 pts = 2.0)
        return value * self.resolve_point_size(epic=epic)

    def working_order_clearance_points(self, order_type: OrderType) -> tuple[float, float]:
        """Return (min_clearance, max_clearance) in **price** units vs dealable quote."""
        rules_min = self.resolve_min_stop_or_limit_distance()
        rules_max = self.resolve_max_stop_or_limit_distance()
        mid = float(self.ledger.last_price or 0.0)
        if 0 < mid < 50:
            # FX: use IG min (POINTS→price). Do not invent a large mid% floor —
            # that deferred mean-reversion TPs as "through market" (e.g. SELL
            # entry above mid with TP a few pips below bid).
            min_c = max(rules_min, self.resolve_point_size() * 2.0)
            max_c = rules_max if rules_max > 0 else mid * 0.5
        else:
            floor = (
                _DEFAULT_STOP_CLEARANCE_POINTS
                if order_type == OrderType.STOP
                else _DEFAULT_LIMIT_CLEARANCE_POINTS
            )
            min_c = max(rules_min, floor)
            max_c = rules_max if rules_max > 0 else _DEFAULT_MAX_CLEARANCE_POINTS
        if max_c < min_c:
            max_c = min_c
        return min_c, max_c

    def apply_working_order_clearance(
        self,
        order: WorkingOrder,
        *,
        limit_level: float | None = None,
    ) -> tuple[float, float | None, list[str]]:
        """Clamp level / attached TP into IG min–max distance from bid/offer.

        Returns (level, limit_level, notes). Mutates ``order.level`` when adjusted.
        """
        notes: list[str] = []
        bid, offer = self.dealable_quote()
        if bid <= 0 or offer <= 0:
            return float(order.level), limit_level, notes
        min_c, max_c = self.working_order_clearance_points(order.type)
        level = float(order.level)

        def _clamp_below(anchor: float, raw: float, label: str) -> float:
            """Resting level must sit in [anchor-max_c, anchor-min_c]."""
            lo, hi = anchor - max_c, anchor - min_c
            out = raw
            if out > hi:
                notes.append(f"{label} {raw}->{hi:.1f} (too close to {anchor:.1f})")
                out = hi
            elif out < lo:
                notes.append(f"{label} {raw}->{lo:.1f} (beyond max {max_c:.1f})")
                out = lo
            return out

        def _clamp_above(anchor: float, raw: float, label: str) -> float:
            """Resting level must sit in [anchor+min_c, anchor+max_c]."""
            lo, hi = anchor + min_c, anchor + max_c
            out = raw
            if out < lo:
                notes.append(f"{label} {raw}->{lo:.1f} (too close to {anchor:.1f})")
                out = lo
            elif out > hi:
                notes.append(f"{label} {raw}->{hi:.1f} (beyond max {max_c:.1f})")
                out = hi
            return out

        if order.type == OrderType.STOP and order.side == Side.BUY:
            level = _clamp_above(offer, level, "BUY STOP")
        elif order.type == OrderType.STOP and order.side == Side.SELL:
            level = _clamp_below(bid, level, "SELL STOP")
        elif order.type == OrderType.LIMIT and order.side == Side.BUY:
            level = _clamp_below(bid, level, "BUY LIMIT")
        elif order.type == OrderType.LIMIT and order.side == Side.SELL:
            level = _clamp_above(offer, level, "SELL LIMIT")

        level = self.snap_level(level)
        order.level = level
        if order.id and order.id in self.ledger.working_orders:
            self.ledger.working_orders[order.id].level = level

        out_tp = limit_level
        if out_tp is not None:
            tp = float(out_tp)
            tp_min, tp_max = self.working_order_clearance_points(OrderType.LIMIT)
            mid = float(self.ledger.last_price or level or 0)
            fx = 0 < mid < 50 or 0 < level < 50
            if fx:
                # FX working-order TP is sent as limitDistance from *entry* (IG UI).
                # Do not require TP to clear the live bid/offer — that blocked
                # mean-reversion brackets the platform accepts manually.
                if order.side == Side.BUY:
                    need = level + tp_min
                    if tp < need:
                        notes.append(
                            f"omit_tp_attach {tp} (need>={need:.5f} entry={level:.5f})"
                        )
                        out_tp = None
                    else:
                        hi = level + tp_max
                        if tp > hi:
                            notes.append(f"clamp TP {tp}->{hi:.5f} (max vs entry)")
                            tp = hi
                        out_tp = self.snap_level(tp)
                else:
                    need = level - tp_min
                    if tp > need:
                        notes.append(
                            f"omit_tp_attach {tp} (need<={need:.5f} entry={level:.5f})"
                        )
                        out_tp = None
                    else:
                        lo = level - tp_max
                        if tp < lo:
                            notes.append(f"clamp TP {tp}->{lo:.5f} (max vs entry)")
                            tp = lo
                        out_tp = self.snap_level(tp)
            elif order.side == Side.BUY:
                # Indices: IG still validates absolute limitLevel vs live quote.
                need = max(level + tp_min, offer + tp_min)
                if tp < need:
                    notes.append(
                        f"omit_tp_attach {tp} (need>={need:.5f} offer={offer:.5f})"
                    )
                    out_tp = None
                else:
                    hi = offer + tp_max
                    if tp > hi:
                        notes.append(f"clamp TP {tp}->{hi:.5f} (max vs offer)")
                        tp = hi
                    out_tp = self.snap_level(tp)
            else:
                need = min(level - tp_min, bid - tp_min)
                if tp > need:
                    notes.append(
                        f"omit_tp_attach {tp} (need<={need:.5f} bid={bid:.5f})"
                    )
                    out_tp = None
                else:
                    lo = bid - tp_max
                    if tp < lo:
                        notes.append(f"clamp TP {tp}->{lo:.5f} (max vs bid)")
                        tp = lo
                    out_tp = self.snap_level(tp)
        return level, out_tp, notes

    def search_markets(self, search_term: str) -> list[dict[str, Any]]:
        """GET /markets?searchTerm= — light market rows (epic, instrumentName, …)."""
        term = (search_term or "").strip()
        if not term or not self._cst:
            return []
        url = f"{self.base_url}/markets"
        resp = self._client.get(
            url,
            headers=self._headers(version="1"),
            params={"searchTerm": term},
        )
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="search_markets", url=url))
        payload = resp.json() if resp.content else {}
        rows = (payload or {}).get("markets") or []
        return [r for r in rows if isinstance(r, dict)]

    def market_currency_codes(self, *, epic: str | None = None) -> list[str]:
        market = self.get_market(epic)
        instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
        codes: list[str] = []
        for row in instrument.get("currencies") or []:
            if isinstance(row, dict) and row.get("code"):
                code = str(row["code"]).strip().upper()
                if code and code not in codes:
                    codes.append(code)
        return codes

    def epic_product_hint(self, epic: str | None = None, *, market: dict[str, Any] | None = None) -> str:
        """
        Rough product family for an epic: SPREADBET | CFD | UNKNOWN.

        Do NOT treat ``DAILY`` alone as spread-bet — IG's France 40 Cash CFD epic is
        ``IX.D.CAC.DAILY.IP`` (daily-funded CFD). Prefer explicit instrument tags, then
        market currencies (GBP-only → spread bet; EUR/USD → CFD).
        """
        ep = (epic or self.config.epic or "").strip().upper()
        parts = ep.split(".")
        tag = parts[3] if len(parts) >= 4 else ""
        if tag in ("CFS", "IFS", "IDF", "IFA", "CASH", "CFD") or any(
            t in ep for t in (".CFS.", ".IFS.", ".IDF.", ".CASH.", ".CFD.")
        ):
            return "CFD"
        allowed: list[str] = []
        if market is not None:
            instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
            for row in instrument.get("currencies") or []:
                if isinstance(row, dict) and row.get("code"):
                    code = str(row["code"]).strip().upper()
                    if code and code not in allowed:
                        allowed.append(code)
        else:
            try:
                allowed = self.market_currency_codes(epic=epic)
            except Exception:
                allowed = []
        if allowed == ["GBP"]:
            return "SPREADBET"
        if any(c in allowed for c in ("EUR", "USD")) and "GBP" not in allowed:
            return "CFD"
        # DAILY/TODAY without currency proof: UNKNOWN (compatible) — not SPREADBET.
        return "UNKNOWN"

    def epic_compatible_with_account(
        self, *, epic: str | None = None, account_type: str | None = None
    ) -> bool:
        """False when CFD account + spread-bet epic (or the reverse)."""
        acc = (account_type or self.resolve_account_type() or "").strip().upper()
        hint = self.epic_product_hint(epic)
        if not acc or hint == "UNKNOWN":
            return True
        if acc == "CFD" and hint == "SPREADBET":
            return False
        if acc == "SPREADBET" and hint == "CFD":
            return False
        return True

    def find_compatible_epic(
        self,
        *,
        search_terms: list[str] | None = None,
        account_type: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """
        Search IG markets for an epic matching the account product type.

        Returns (chosen_epic_or_None, candidate_epics_seen).
        """
        acc = (account_type or self.resolve_account_type() or "").strip().upper()
        want_sb = acc == "SPREADBET"
        want_cfd = acc == "CFD"
        terms = search_terms or ["France 40", "CAC 40", "CAC40"]
        seen: list[str] = []
        candidates: list[tuple[int, str]] = []
        for term in terms:
            try:
                rows = self.search_markets(term)
            except IgApiError:
                continue
            for row in rows:
                ep = str(row.get("epic") or "").strip()
                if not ep or ep in seen:
                    continue
                seen.append(ep)
                hint = self.epic_product_hint(ep)
                name = str(row.get("instrumentName") or "").upper()
                status = str(row.get("marketStatus") or "").upper()
                score = 0
                if want_cfd and hint == "CFD":
                    score += 10
                if want_sb and hint == "SPREADBET":
                    score += 10
                if want_cfd and hint == "SPREADBET":
                    score -= 20
                if want_sb and hint == "CFD":
                    score -= 20
                if "FRANCE 40" in name or "CAC" in name:
                    score += 3
                if status == "TRADEABLE":
                    score += 2
                if score > 0:
                    candidates.append((score, ep))
        candidates.sort(key=lambda t: (-t[0], t[1]))
        chosen = candidates[0][1] if candidates else None
        return chosen, seen

    def resolve_order_currency(self, *, epic: str | None = None) -> str:
        """Pick a currency IG accepts for working orders on this epic/account."""
        configured = str(getattr(self.config, "ig_currency", "") or "").strip().upper()
        if configured:
            return configured
        allowed = self.market_currency_codes(epic=epic)
        account_ccy = ""
        try:
            acc = self.get_active_account()
            account_ccy = str(
                acc.get("currency") or acc.get("preferredCurrency") or ""
            ).strip().upper()
        except IgApiError:
            account_ccy = ""
        if account_ccy and (not allowed or account_ccy in allowed):
            return account_ccy
        for prefer in ("EUR", "USD", "GBP"):
            if prefer in allowed:
                return prefer
        return allowed[0] if allowed else (account_ccy or "EUR")

    def resolve_order_expiry(self, *, epic: str | None = None) -> str:
        """
        Instrument expiry for working orders.

        Spread bets (DAILY) usually need DFB. Undated CFDs use "-".
        Prefer the market's own expiry; only default when missing.
        """
        market = self.get_market(epic)
        instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
        expiry = str(instrument.get("expiry") or "").strip()
        if expiry and expiry not in ("null", "None"):
            return expiry
        hint = self.epic_product_hint(epic, market=market)
        return "DFB" if hint == "SPREADBET" else "-"

    def resolve_min_deal_size(self, *, epic: str | None = None) -> float:
        market = self.get_market(epic)
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        min_deal = rules.get("minDealSize") or {}
        if isinstance(min_deal, dict) and min_deal.get("value") is not None:
            try:
                return float(min_deal["value"])
            except (TypeError, ValueError):
                pass
        return float(self.config.order_size or 1.0)

    def resolve_price_step(self, *, epic: str | None = None) -> float:
        """Minimum price increment for snapping working-order levels.

        FX: use **pipette** (``point_size / 10``, typically 0.00001). IG's
        ``minStepDistance`` is often 5 POINTS (0.0005) on EURUSD Mini, but the
        web UI and OTC amend path accept 5-decimal prices (e.g. 1.13721). Snapping
        to 0.0005 wrongly collapsed a 2-point hedge nudge (1.13730 → 1.13750).

        Indices: prefer ``minStepDistance`` from dealingRules.
        """
        point = self.resolve_point_size(epic=epic)
        mid = float(self.ledger.last_price or 0.0)
        if mid <= 0:
            market = self.get_market(epic) if self._cst else {}
            snap = (market.get("snapshot") or {}) if isinstance(market, dict) else {}
            try:
                bid = float(snap.get("bid") or 0)
                offer = float(snap.get("offer") or 0)
                if bid > 0 and offer > 0:
                    mid = (bid + offer) / 2.0
            except (TypeError, ValueError):
                mid = 0.0
        if point > 0 and (point <= 0.0001 + 1e-15 or 0 < mid < 10):
            return point / 10.0

        market = self.get_market(epic) if self._cst else {}
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        # Only minStepDistance — never minNormalStopOrLimitDistance (that is a
        # stop/limit clearance, not a tick size).
        row = rules.get("minStepDistance") or {}
        if isinstance(row, dict) and row.get("value") is not None:
            try:
                step = float(row["value"])
                if step > 0:
                    unit = str(row.get("unit") or "POINTS").strip().upper()
                    if unit == "POINTS":
                        return step * self.resolve_point_size(epic=epic)
                    return step
            except (TypeError, ValueError):
                pass
        return 0.1 if point <= 0 else point

    def snap_level(self, level: float, *, epic: str | None = None) -> float:
        """Round a price level to the instrument step (avoids float noise like …000001)."""
        from decimal import Decimal, ROUND_HALF_UP

        step = self.resolve_price_step(epic=epic)
        if step <= 0:
            return float(level)
        d_level = Decimal(str(level))
        d_step = Decimal(str(step))
        snapped = (d_level / d_step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * d_step
        # Keep a sensible number of decimals for JSON (no binary float junk).
        return float(snapped)

    def place_order(
        self,
        order: WorkingOrder,
        *,
        currency: str | None = None,
        limit_level: float | None = None,
        stop_level: float | None = None,
    ) -> WorkingOrder:
        placed = self.ledger.place_order(order)
        if self.dry_run or not self._cst:
            logger.info(
                "IG dry-run place %s limit_level=%s stop_level=%s",
                placed.to_dict(),
                limit_level,
                stop_level,
            )
            return placed
        return self.push_working_order(
            placed,
            currency=currency,
            limit_level=limit_level,
            stop_level=stop_level,
        )

    def push_working_order(
        self,
        order: WorkingOrder,
        *,
        currency: str | None = None,
        limit_level: float | None = None,
        stop_level: float | None = None,
    ) -> WorkingOrder:
        """Submit an already-ledgered working order to IG (no second ledger place)."""
        level, limit_level, clearance_notes = self.apply_working_order_clearance(
            order, limit_level=limit_level
        )
        if clearance_notes:
            logger.info(
                "IG WO clearance order=%s %s",
                order.id or "—",
                "; ".join(clearance_notes),
            )
        if self.dry_run or not self._cst:
            logger.info(
                "IG dry-run push %s limit_level=%s stop_level=%s clearance=%s",
                order.to_dict(),
                limit_level,
                stop_level,
                clearance_notes,
            )
            self._note_ig_result(
                action="place_working_order",
                deal_status="DRY_RUN",
                deal_id=order.deal_id or "",
                level=level,
            )
            return order
        body = self._ig_working_order_body(
            order,
            currency=currency,
            limit_level=limit_level,
            stop_level=stop_level,
        )
        # Keep snapped/widened level on the order object used for logging.
        order.level = float(body.get("level") or level)
        resp = self._client.post(
            f"{self.base_url}/workingorders/otc",
            headers=self._headers(version="2"),
            json=body,
        )
        if resp.is_error:
            raise IgApiError(
                format_ig_http_error(resp, action="place_working_order", url=str(resp.request.url))
            )
        deal = resp.json() if resp.content else {}
        order.client_ref = str((deal or {}).get("dealReference") or order.client_ref)
        deal_status = ""
        reason = ""
        if order.client_ref:
            confirmed = self.confirm_deal(order.client_ref)
            deal_status = str(confirmed.get("dealStatus") or "").upper()
            reason = str(confirmed.get("reason") or "").strip()
            order.deal_id = str(confirmed.get("dealId") or "")
            if order.id in self.ledger.working_orders:
                self.ledger.working_orders[order.id].client_ref = order.client_ref
                self.ledger.working_orders[order.id].deal_id = order.deal_id
                self.ledger.working_orders[order.id].level = order.level
            bid, offer = self.dealable_quote()
            self._note_ig_result(
                action="place_working_order",
                deal_reference=order.client_ref,
                deal_id=order.deal_id,
                deal_status=deal_status or "UNKNOWN",
                reason=reason,
                level=order.level,
                # FX uses limitDistance; indices may use limitLevel. Either means TP went out.
                limit_level=body.get("limitLevel"),
                limit_distance=body.get("limitDistance"),
                tp_attached=(
                    body.get("limitLevel") is not None
                    or body.get("limitDistance") is not None
                ),
                bid=bid,
                offer=offer,
                clearance="; ".join(clearance_notes) if clearance_notes else None,
            )
            if deal_status and deal_status != "ACCEPTED":
                # Do not leave phantom dealIds — they show up as "Dropped local orders".
                rejected_deal = order.deal_id
                order.deal_id = ""
                if order.id in self.ledger.working_orders:
                    self.ledger.working_orders[order.id].deal_id = ""
                raise IgApiError(
                    "IG working order rejected: "
                    f"dealStatus={deal_status or '—'} reason={reason or '—'} "
                    f"dealId={rejected_deal or '—'} "
                    f"currency={body.get('currencyCode')} expiry={body.get('expiry')} "
                    f"forceOpen={body.get('forceOpen')} "
                    f"limitLevel={body.get('limitLevel')} "
                    f"bid={bid} offer={offer} "
                    f"({order.side.value} {order.type.value} @ {order.level}) "
                    f"confirm={confirmed}"
                )
        else:
            self._note_ig_result(
                action="place_working_order",
                deal_id=order.deal_id or "",
                deal_status="NO_REF",
            )
        return order

    def confirm_deal(self, deal_reference: str, *, retries: int = 8, pause: float = 0.35) -> dict[str, Any]:
        """Poll GET /confirms/{dealReference} until dealStatus is present."""
        ref = (deal_reference or "").strip()
        if not ref or not self._cst:
            return {}
        url = f"{self.base_url}/confirms/{ref}"
        last: dict[str, Any] = {}
        for _attempt in range(max(1, retries)):
            resp = self._client.get(url, headers=self._headers(version="1"))
            if resp.status_code == 404:
                time.sleep(pause)
                continue
            if resp.is_error:
                raise IgApiError(format_ig_http_error(resp, action="confirm_deal", url=url))
            payload = resp.json() if resp.content else {}
            last = payload if isinstance(payload, dict) else {}
            # Wait until IG has finished deciding (ACCEPTED or REJECTED).
            if last.get("dealStatus"):
                return last
            time.sleep(pause)
        return last

    def list_working_orders(self) -> list[dict[str, Any]]:
        """Return IG working-order rows for the active account."""
        if not self._cst:
            return []
        url = f"{self.base_url}/workingorders"
        resp = self._client.get(url, headers=self._headers(version="2"))
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="list_working_orders", url=url))
        payload = resp.json() if resp.content else {}
        rows = (payload or {}).get("workingOrders") or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            data = row.get("workingOrderData") or row
            if isinstance(data, dict):
                out.append(data)
        return out

    def list_open_positions(self, *, epic: str | None = None) -> list[dict[str, Any]]:
        """
        GET /positions — open deals for the active account.

        Each item: {deal_id, epic, side, size, level, currency, raw}.
        When ``epic`` is set (default: config.epic), only matching rows are returned.
        """
        if not self._cst:
            return []
        url = f"{self.base_url}/positions"
        resp = self._client.get(url, headers=self._headers(version="2"))
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="list_positions", url=url))
        payload = resp.json() if resp.content else {}
        rows = (payload or {}).get("positions") or []
        want = (epic if epic is not None else self.config.epic or "").strip()
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            market = row.get("market") or {}
            pos = row.get("position") or {}
            if not isinstance(market, dict):
                market = {}
            if not isinstance(pos, dict):
                pos = {}
            row_epic = str(market.get("epic") or pos.get("epic") or "").strip()
            if want and row_epic and row_epic != want:
                continue
            bid = _float_or_none(market.get("bid"))
            offer = _float_or_none(
                market.get("offer") if market.get("offer") is not None else market.get("ask")
            )
            if bid is not None and bid > 0:
                self.last_dealable_bid = bid
            if offer is not None and offer > 0:
                self.last_dealable_offer = offer
            direction = str(pos.get("direction") or "").upper()
            side = Side.BUY if direction == "BUY" else Side.SELL
            try:
                size = float(pos.get("size") or 0)
            except (TypeError, ValueError):
                size = 0.0
            try:
                level = float(pos.get("level") or pos.get("openLevel") or 0)
            except (TypeError, ValueError):
                level = 0.0
            stop_level: float | None = None
            limit_level: float | None = None
            for key, dest in (("stopLevel", "stop"), ("limitLevel", "limit")):
                raw_lvl = pos.get(key)
                if raw_lvl is None:
                    continue
                try:
                    val = float(raw_lvl)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    if dest == "stop":
                        stop_level = val
                    else:
                        limit_level = val
            out.append(
                {
                    "deal_id": str(pos.get("dealId") or ""),
                    "epic": row_epic,
                    "side": side,
                    "size": size,
                    "level": level,
                    "stop_level": stop_level,
                    "limit_level": limit_level,
                    "currency": str(pos.get("currency") or ""),
                    "raw": row,
                }
            )
        return out

    def ig_net_size(self, *, epic: str | None = None) -> float:
        """Signed IG exposure for epic: +BUY −SELL."""
        net = 0.0
        for row in self.list_open_positions(epic=epic):
            size = float(row.get("size") or 0)
            side = row.get("side")
            if side == Side.BUY:
                net += size
            else:
                net -= size
        return net

    def open_market_position(
        self,
        side: Side,
        size: float,
        *,
        role: LegRole = LegRole.HEDGE,
        currency: str | None = None,
    ) -> str:
        """
        Open a market position (forceOpen) and mirror into the local ledger.

        Live: POST /positions/otc. Paper/dry_run: ledger only.
        """
        qty = abs(float(size))
        if qty <= 0:
            raise IgApiError("open_market_position requires size > 0")
        if self.dry_run or not self._cst:
            logger.info(
                "IG dry-run market open %s size=%s role=%s", side.value, qty, role.value
            )
            return self.ledger.market_open(side, qty, role=role)

        ccy = (currency or self.resolve_order_currency()).strip().upper()
        expiry = self.resolve_order_expiry()
        body = {
            "epic": self.config.epic,
            "expiry": expiry,
            "direction": "BUY" if side == Side.BUY else "SELL",
            "size": qty,
            "orderType": "MARKET",
            "currencyCode": ccy,
            "forceOpen": True,
            "guaranteedStop": False,
        }
        url = f"{self.base_url}/positions/otc"
        resp = self._client.post(url, headers=self._headers(version="2"), json=body)
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="open_market_position", url=url))
        deal = resp.json() if resp.content else {}
        deal_ref = str((deal or {}).get("dealReference") or "").strip()
        fill = float(self.ledger.last_price or 0)
        deal_id = ""
        if deal_ref:
            confirmed = self.confirm_deal(deal_ref)
            deal_status = str(confirmed.get("dealStatus") or "").upper()
            reason = str(confirmed.get("reason") or "").strip()
            deal_id = str(confirmed.get("dealId") or "").strip()
            if deal_status and deal_status != "ACCEPTED":
                raise IgApiError(
                    "IG market open rejected: "
                    f"dealStatus={deal_status or '—'} reason={reason or '—'} "
                    f"confirm={confirmed}"
                )
            try:
                if confirmed.get("level") is not None:
                    fill = float(confirmed["level"])
            except (TypeError, ValueError):
                pass
        if fill <= 0:
            half = abs(self.config.spread_points) / 2.0
            fill = (
                float(self.ledger.last_price or 0) + half
                if side == Side.BUY
                else float(self.ledger.last_price or 0) - half
            )
        leg = self.ledger._open_leg(side, qty, fill, role, deal_id=deal_id)
        logger.info(
            "IG market open accepted %s size=%s dealRef=%s dealId=%s leg=%s",
            side.value,
            qty,
            deal_ref or "—",
            deal_id or "—",
            leg.id,
        )
        return leg.id

    def cancel_working_order(self, deal_id: str) -> dict[str, Any]:
        """Cancel OTC working order (IG: POST + ``_method: DELETE``, not bare DELETE).

        A plain HTTP DELETE often returns ``validation.null-not-allowed.request`` on
        IG's public gateway. The official trading-ig client tunnels deletes as POST
        with an ``_method: DELETE`` header and an empty JSON body.
        """
        did = (deal_id or "").strip()
        if not did:
            raise IgApiError("IG cancel requires dealId")
        if self.dry_run or not self._cst:
            logger.info("IG dry-run cancel dealId=%s", did)
            self._note_ig_result(
                action="cancel_working_order", deal_id=did, deal_status="DRY_RUN"
            )
            return {"dealId": did, "dry_run": True}
        url = f"{self.base_url}/workingorders/otc/{did}"
        last_exc: IgApiError | None = None
        for version in ("2", "1"):
            headers = dict(self._headers(version=version))
            headers["_method"] = "DELETE"
            resp = self._client.post(url, headers=headers, json={})
            if not resp.is_error:
                payload = resp.json() if resp.content else {}
                out = payload if isinstance(payload, dict) else {"raw": payload}
                self._note_ig_result(
                    action="cancel_working_order",
                    deal_id=did,
                    deal_reference=str(out.get("dealReference") or ""),
                    deal_status="ACCEPTED",
                )
                return out
            last_exc = IgApiError(
                format_ig_http_error(resp, action="cancel_working_order", url=url)
            )
            code = ""
            try:
                code = str((resp.json() or {}).get("errorCode") or "")
            except Exception:
                pass
            # Already gone — treat as success so the local order book can drop it.
            if "not.found" in code.lower() or "cannot.be.found" in code.lower():
                logger.info("IG cancel dealId=%s already gone (%s)", did, code or "not.found")
                self._note_ig_result(
                    action="cancel_working_order",
                    deal_id=did,
                    deal_status="ALREADY_GONE",
                    reason=code or "not.found",
                )
                return {"dealId": did, "already_gone": True}
            # Try the other API version before giving up.
        assert last_exc is not None
        raise last_exc

    def amend_working_order_by_deal_id(
        self,
        deal_id: str,
        *,
        order_type: OrderType | str,
        level: float,
        limit_level: float | None = None,
        stop_level: float | None = None,
    ) -> float:
        """PUT /workingorders/otc/{dealId} — level amend; returns snapped level.

        Re-send ``limitLevel`` / ``stopLevel`` when provided so attached TP/SL
        on an ENTRY are not cleared by IG.
        """
        did = (deal_id or "").strip()
        if not did or did.startswith("attached:"):
            raise IgApiError("IG amend requires a standalone dealId")
        snapped = self.snap_level(float(level))
        if self.dry_run or not self._cst:
            logger.info(
                "IG dry-run amend dealId=%s level=%s limit=%s stop=%s",
                did,
                snapped,
                limit_level,
                stop_level,
            )
            self._note_ig_result(
                action="amend_working_order",
                deal_id=did,
                deal_status="DRY_RUN",
                level=snapped,
            )
            return snapped
        otype = (
            order_type
            if isinstance(order_type, OrderType)
            else OrderType.STOP
            if "STOP" in str(order_type).upper()
            else OrderType.LIMIT
        )
        body: dict[str, Any] = {
            "type": "LIMIT" if otype == OrderType.LIMIT else "STOP",
            "level": snapped,
            "timeInForce": "GOOD_TILL_CANCELLED",
        }
        # FX: prefer limitDistance (same as place). Absolute limitLevel is often
        # rejected vs live quote when attaching TP on a resting entry.
        if limit_level is not None:
            self._attach_working_order_tp(
                body, level=snapped, limit_level=float(limit_level)
            )
        if stop_level is not None:
            body["stopLevel"] = self.snap_level(float(stop_level))
        url = f"{self.base_url}/workingorders/otc/{did}"
        last_exc: IgApiError | None = None
        for version in ("2", "1"):
            resp = self._client.put(url, headers=self._headers(version=version), json=body)
            if not resp.is_error:
                payload = resp.json() if resp.content else {}
                ref = ""
                if isinstance(payload, dict):
                    ref = str(payload.get("dealReference") or "")
                logger.info(
                    "IG amend accepted dealId=%s level=%s", did, snapped
                )
                self._note_ig_result(
                    action="amend_working_order",
                    deal_id=did,
                    deal_reference=ref,
                    deal_status="ACCEPTED",
                    level=snapped,
                    limit_level=body.get("limitLevel"),
                    limit_distance=body.get("limitDistance"),
                    tp_attached=(
                        body.get("limitLevel") is not None
                        or body.get("limitDistance") is not None
                    ),
                )
                return snapped
            last_exc = IgApiError(
                format_ig_http_error(resp, action="amend_working_order", url=url)
            )
        assert last_exc is not None
        raise last_exc

    def amend_order(
        self, order_id: str, *, level: float, size: float | None = None
    ) -> WorkingOrder:
        """Amend local ledger; level PUT to IG when dealId exists.

        Size changes on the live path are handled by the scheduler mirror
        (cancel + re-place). This method updates ledger size when provided and
        only PUTs level to IG.
        """
        order = self.ledger.amend_order(order_id, level=level, size=size)
        if self.dry_run or not self._cst:
            return order
        deal_id = (order.deal_id or "").strip()
        snapped = self.snap_level(float(level))
        order.level = snapped
        if order.id in self.ledger.working_orders:
            self.ledger.working_orders[order.id].level = snapped
        if not deal_id or deal_id.startswith("attached:"):
            logger.warning(
                "IG amend skipped (no standalone dealId) for local order %s", order_id
            )
            return order
        limit_level: float | None = None
        stop_level: float | None = None
        try:
            for raw in self.list_working_orders() or []:
                if not isinstance(raw, dict):
                    continue
                data = raw.get("workingOrderData") if isinstance(
                    raw.get("workingOrderData"), dict
                ) else raw
                if not isinstance(data, dict):
                    continue
                if str(data.get("dealId") or "").strip() != deal_id:
                    continue
                if data.get("limitLevel") is not None:
                    limit_level = float(data["limitLevel"])
                if data.get("stopLevel") is not None:
                    stop_level = float(data["stopLevel"])
                break
        except Exception:
            logger.exception("IG amend: failed to load remote WO for %s", deal_id)
        self.amend_working_order_by_deal_id(
            deal_id,
            order_type=order.type,
            level=snapped,
            limit_level=limit_level,
            stop_level=stop_level,
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        order = self.ledger.working_orders.get(order_id)
        deal_id = (order.deal_id if order else "") or ""
        self.ledger.cancel_order(order_id)
        if self.dry_run or not self._cst:
            return
        if not deal_id or deal_id.startswith("attached:"):
            # Best-effort: match by client_ref via open working orders
            if not deal_id:
                logger.warning("IG cancel skipped (no dealId) for local order %s", order_id)
            return
        self.cancel_working_order(deal_id)

    def update_position_protection(
        self,
        deal_id: str,
        *,
        stop_level: float | None = None,
        limit_level: float | None = None,
    ) -> dict[str, Any]:
        """PUT /positions/otc/{dealId} — attach/amend stop/limit on an open deal."""
        did = (deal_id or "").strip()
        if not did:
            raise IgApiError("update_position_protection requires dealId")
        body: dict[str, Any] = {}
        if stop_level is not None:
            body["stopLevel"] = self.snap_level(float(stop_level))
        if limit_level is not None:
            body["limitLevel"] = self.snap_level(float(limit_level))
        if not body:
            return {}
        if self.dry_run or not self._cst:
            logger.info("IG dry-run position protection dealId=%s %s", did, body)
            return {"dealId": did, "dry_run": True, **body}
        url = f"{self.base_url}/positions/otc/{did}"
        # Avoid stamping a prior WO reject onto this attach via last_ig_result.
        self.last_ig_result = {}
        last_exc: IgApiError | None = None
        for version in ("2", "1"):
            resp = self._client.put(url, headers=self._headers(version=version), json=body)
            if not resp.is_error:
                payload = resp.json() if resp.content else {}
                if not isinstance(payload, dict):
                    payload = {"raw": payload}
                deal_ref = str(payload.get("dealReference") or "").strip()
                if deal_ref:
                    confirmed = self.confirm_deal(deal_ref)
                    deal_status = str(confirmed.get("dealStatus") or "").upper()
                    reason = str(confirmed.get("reason") or "").strip()
                    self._note_ig_result(
                        action="update_position_protection",
                        deal_reference=deal_ref,
                        deal_id=did,
                        deal_status=deal_status or "UNKNOWN",
                        reason=reason,
                        limit_level=body.get("limitLevel"),
                        stop_level=body.get("stopLevel"),
                    )
                    if deal_status and deal_status != "ACCEPTED":
                        raise IgApiError(
                            "IG position protection rejected: "
                            f"dealStatus={deal_status or '—'} reason={reason or '—'} "
                            f"dealId={did} confirm={confirmed}"
                        )
                else:
                    self._note_ig_result(
                        action="update_position_protection",
                        deal_id=did,
                        deal_status="NO_REF",
                        limit_level=body.get("limitLevel"),
                        stop_level=body.get("stopLevel"),
                    )
                return payload
            last_exc = IgApiError(
                format_ig_http_error(resp, action="update_position_protection", url=url)
            )
        assert last_exc is not None
        raise last_exc

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
            self.market_close(position_id)
            return
        # Prefer attaching/updating limit on the IG deal when we have dealId.
        deal_id = (leg.deal_id or "").strip()
        if deal_id and not self.dry_run and self._cst:
            try:
                self.update_position_protection(deal_id, limit_level=float(level))
                from chatbot.trader.models import OrderPurpose, attached_deal_id

                self.place_order(
                    WorkingOrder(
                        id="",
                        type=OrderType.LIMIT,
                        side=Side.SELL if leg.side == Side.BUY else Side.BUY,
                        level=float(level),
                        size=leg.size,
                        purpose=OrderPurpose.TP,
                        position_id=position_id,
                        deal_id=attached_deal_id(deal_id, OrderPurpose.TP),
                    )
                )
                return
            except Exception:
                logger.exception(
                    "IG attach TP failed dealId=%s; falling back to local WO only",
                    deal_id,
                )
        close_side = Side.SELL if leg.side == Side.BUY else Side.BUY
        from chatbot.trader.models import OrderPurpose

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

    def market_open(self, side: Side, size: float, *, role: LegRole = LegRole.PRIMARY) -> str:
        return self.open_market_position(side, size, role=role)

    def market_close(self, position_id: str) -> None:
        """Close a leg at market. Live: DELETE /positions/otc (via POST + _method); paper: ledger."""
        leg = self.ledger.positions.get(position_id)
        if not leg:
            return
        if self.dry_run or not self._cst:
            self.ledger.market_close(position_id)
            return
        deal_id = (leg.deal_id or "").strip()
        if not deal_id:
            raise IgApiError(
                f"market_close requires IG dealId on leg {position_id} "
                "(reconcile/adopt first)"
            )
        close_side = Side.SELL if leg.side == Side.BUY else Side.BUY
        qty = abs(float(leg.size))
        # IG closes positions via DELETE /positions/otc (v1); since DELETE bodies
        # are unreliable, IG documents POST + "_method: DELETE" header. A plain
        # POST is treated as "open position" and fails validation
        # (null-not-allowed.request.guaranteedStop). Close by dealId only —
        # epic/expiry are for close-by-market and must not be combined.
        body = {
            "dealId": deal_id,
            "direction": "BUY" if close_side == Side.BUY else "SELL",
            "size": qty,
            "orderType": "MARKET",
        }
        url = f"{self.base_url}/positions/otc"
        headers = self._headers(version="1")
        headers["_method"] = "DELETE"
        resp = self._client.post(url, headers=headers, json=body)
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="market_close", url=url))
        deal = resp.json() if resp.content else {}
        deal_ref = str((deal or {}).get("dealReference") or "").strip()
        exit_px = self.ledger.market_close_fill_price(leg)
        if deal_ref:
            confirmed = self.confirm_deal(deal_ref)
            deal_status = str(confirmed.get("dealStatus") or "").upper()
            reason = str(confirmed.get("reason") or "").strip()
            if deal_status and deal_status != "ACCEPTED":
                raise IgApiError(
                    "IG market close rejected: "
                    f"dealStatus={deal_status or '—'} reason={reason or '—'} "
                    f"confirm={confirmed}"
                )
            try:
                if confirmed.get("level") is not None:
                    exit_px = float(confirmed["level"])
            except (TypeError, ValueError):
                pass
        self.ledger.close_position(position_id, exit_px)
        logger.info(
            "IG market close accepted leg=%s dealId=%s exit=%s",
            position_id,
            deal_id,
            exit_px,
        )

    def submit_working_order_raw(
        self,
        body: dict[str, Any],
        *,
        version: str = "2",
    ) -> dict[str, Any]:
        """POST /workingorders/otc with an explicit body; return confirm + meta.

        Used by DEMO diagnostics (and tests). Does not touch the ledger.
        """
        if self.dry_run or not self._cst:
            return {
                "dealStatus": "DRY_RUN",
                "reason": "",
                "dealId": "",
                "dealReference": "",
                "body": body,
                "version": version,
            }
        resp = self._client.post(
            f"{self.base_url}/workingorders/otc",
            headers=self._headers(version=version),
            json=body,
        )
        if resp.is_error:
            raise IgApiError(
                format_ig_http_error(resp, action="place_working_order", url=str(resp.request.url))
            )
        deal = resp.json() if resp.content else {}
        deal_ref = str((deal or {}).get("dealReference") or "").strip()
        confirmed: dict[str, Any] = {}
        if deal_ref:
            confirmed = self.confirm_deal(deal_ref)
        out = dict(confirmed) if isinstance(confirmed, dict) else {}
        out.setdefault("dealReference", deal_ref)
        out["body"] = body
        out["version"] = version
        return out

    def _ig_working_order_body(
        self,
        order: WorkingOrder,
        *,
        currency: str | None = None,
        limit_level: float | None = None,
        stop_level: float | None = None,
    ) -> dict[str, Any]:
        ccy = (currency or self.resolve_order_currency()).strip().upper()
        expiry = self.resolve_order_expiry()
        level = self.snap_level(float(order.level))
        order.level = level
        body: dict[str, Any] = {
            "epic": self.config.epic,
            "expiry": expiry,
            "direction": "BUY" if order.side == Side.BUY else "SELL",
            "size": float(order.size),
            "level": level,
            "type": "LIMIT" if order.type == OrderType.LIMIT else "STOP",
            "currencyCode": ccy,
            "timeInForce": "GOOD_TILL_CANCELLED",
            "guaranteedStop": False,
            # IG requires forceOpen=true for LIMIT/STOP working orders that open.
            "forceOpen": True,
        }
        if limit_level is not None:
            self._attach_working_order_tp(body, level=level, limit_level=float(limit_level))
        if stop_level is not None:
            body["stopLevel"] = self.snap_level(float(stop_level))
        return body

    def _attach_working_order_tp(
        self,
        body: dict[str, Any],
        *,
        level: float,
        limit_level: float,
    ) -> None:
        """Attach take-profit on a working order (limitLevel or limitDistance).

        FX (mid/level &lt; 50): always ``limitDistance`` in POINTS from the entry —
        same as IG web UI ("Limit Distance 19.7"). Absolute ``limitLevel`` is
        validated against the live quote and wrongly deferred mean-reversion TPs.

        Index CFDs with POINTS-capped max: also prefer distance. Otherwise
        absolute ``limitLevel``.
        """
        tp = self.snap_level(limit_level)
        market = self.get_market() if self._cst else {}
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        mx = rules.get("maxStopOrLimitDistance") or {}
        mn = rules.get("minNormalStopOrLimitDistance") or {}
        try:
            max_v = float(mx.get("value") or 0)
        except (TypeError, ValueError):
            max_v = 0.0
        try:
            min_v = float(mn.get("value") or 0)
        except (TypeError, ValueError):
            min_v = 0.0
        max_u = str(mx.get("unit") or "POINTS").upper()
        fx = level < 50
        if fx:
            pip = self.resolve_point_size()
            if pip <= 0:
                pip = 0.0001
            raw_dist = abs(float(tp) - float(level)) / pip
            lo = max(min_v, 2.0) if min_v > 0 else 2.0
            if max_u == "PERCENTAGE" and max_v > 0:
                hi = (abs(float(level)) * max_v / 100.0) / pip
            elif max_u == "POINTS" and max_v > 0:
                hi = max_v * 0.95
            else:
                hi = max(raw_dist, lo)
            dist = raw_dist
            if hi > 0 and dist > hi:
                dist = hi
            if dist < lo:
                dist = min(lo, hi) if hi > 0 else lo
            body["limitDistance"] = float(round(dist, 1))
            return
        if max_u == "POINTS" and max_v > 0:
            # Index with POINTS max — distance in index points (1 pt = 1.0 price).
            raw_dist = abs(float(tp) - float(level))
            lo = max(min_v * 2.0 if min_v > 0 else 10.0, 10.0)
            hi = max_v * 0.8 if max_v > lo else max_v
            dist = raw_dist
            if dist > hi > 0:
                dist = hi
            if dist < lo:
                dist = min(lo, hi) if hi > 0 else lo
            body["limitDistance"] = float(round(dist, 2))
            return
        body["limitLevel"] = tp


def _extract_price_allowance(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pull historical-data allowance fields from an IG /prices JSON body."""
    from datetime import datetime, timezone

    meta = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        return None
    raw = meta.get("allowance")
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for key in (
        "remainingAllowance",
        "totalAllowance",
        "allowanceExpiry",
        "remaining",
        "total",
        "expiry",
    ):
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    # Normalize common aliases for status UI.
    if "remaining" not in out and "remainingAllowance" in out:
        out["remaining"] = out["remainingAllowance"]
    if "total" not in out and "totalAllowance" in out:
        out["total"] = out["totalAllowance"]
    if "expiry" not in out and "allowanceExpiry" in out:
        out["expiry"] = out["allowanceExpiry"]
    if out:
        # allowanceExpiry is seconds-remaining *at response time* — stamp for UI countdown.
        out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return out or None


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
        ts = pd.Timestamp(snap)
        # snapshotTimeUTC is UTC; naive snapshotTime is often exchange-local — assume UTC
        # when the UTC field was used, otherwise leave naive for caller to localize.
        if p.get("snapshotTimeUTC") and ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        rows.append(
            {
                "ts": ts,
                "open": mid_o,
                "high": mid_h,
                "low": mid_l,
                "close": mid_c,
                "volume": p.get("lastTradedVolume") or 0,
            }
        )
    if not rows:
        return _EMPTY_OHLC.copy()
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
