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
        if not self._cst:
            return {}
        ep = (epic or self.config.epic or "").strip()
        if not ep:
            return {}
        if not hasattr(self, "_market_cache"):
            self._market_cache: dict[str, dict[str, Any]] = {}
        if ep in self._market_cache:
            return self._market_cache[ep]
        url = f"{self.base_url}/markets/{ep}"
        resp = self._client.get(url, headers=self._headers(version="3"))
        if resp.is_error:
            raise IgApiError(format_ig_http_error(resp, action="markets", url=url))
        payload = resp.json() if resp.content else {}
        market = payload if isinstance(payload, dict) else {}
        self._market_cache[ep] = market
        return market

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

        DAILY.* / GBP-only index epics are typically UK spread bets.
        CFS / IFS / IDF / CASH tags (and EUR/USD currencies) are typically CFDs.
        """
        ep = (epic or self.config.epic or "").strip().upper()
        parts = ep.split(".")
        tag = parts[3] if len(parts) >= 4 else ""
        if tag in ("DAILY", "TODAY") or "DAILY" in ep:
            return "SPREADBET"
        if tag in ("CFS", "IFS", "IDF", "IFA", "CASH", "CFD") or any(
            t in ep for t in (".CFS.", ".IFS.", ".IDF.", ".CASH.", ".CFD.")
        ):
            return "CFD"
        allowed = self.market_currency_codes(epic=epic) if market is None else []
        if market is not None:
            instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
            for row in instrument.get("currencies") or []:
                if isinstance(row, dict) and row.get("code"):
                    code = str(row["code"]).strip().upper()
                    if code and code not in allowed:
                        allowed.append(code)
        if allowed == ["GBP"]:
            return "SPREADBET"
        if any(c in allowed for c in ("EUR", "USD")) and "GBP" not in allowed:
            return "CFD"
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
        """Minimum price increment from dealingRules (fallback 0.1 for indices)."""
        market = self.get_market(epic)
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        for key in ("minStepDistance", "minDealDistance", "minNormalStopOrLimitDistance"):
            row = rules.get(key) or {}
            if isinstance(row, dict) and row.get("value") is not None:
                try:
                    step = float(row["value"])
                    if step > 0:
                        return step
                except (TypeError, ValueError):
                    pass
        return 0.1

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

    def place_order(self, order: WorkingOrder, *, currency: str | None = None) -> WorkingOrder:
        placed = self.ledger.place_order(order)
        if self.dry_run or not self._cst:
            logger.info("IG dry-run place %s", placed.to_dict())
            return placed
        body = self._ig_working_order_body(placed, currency=currency)
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
        placed.client_ref = str((deal or {}).get("dealReference") or placed.client_ref)
        if placed.client_ref:
            confirmed = self.confirm_deal(placed.client_ref)
            deal_status = str(confirmed.get("dealStatus") or "").upper()
            reason = str(confirmed.get("reason") or "").strip()
            placed.deal_id = str(confirmed.get("dealId") or "")
            if placed.id in self.ledger.working_orders:
                self.ledger.working_orders[placed.id].client_ref = placed.client_ref
                self.ledger.working_orders[placed.id].deal_id = placed.deal_id
            if deal_status and deal_status != "ACCEPTED":
                raise IgApiError(
                    "IG working order rejected: "
                    f"dealStatus={deal_status or '—'} reason={reason or '—'} "
                    f"dealId={placed.deal_id or '—'} "
                    f"currency={body.get('currencyCode')} expiry={body.get('expiry')} "
                    f"forceOpen={body.get('forceOpen')} "
                    f"({placed.side.value} {placed.type.value} @ {placed.level}) "
                    f"confirm={confirmed}"
                )
        return placed

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

    def cancel_working_order(self, deal_id: str) -> dict[str, Any]:
        """DELETE /workingorders/otc/{dealId}."""
        did = (deal_id or "").strip()
        if not did:
            raise IgApiError("IG cancel requires dealId")
        if self.dry_run or not self._cst:
            logger.info("IG dry-run cancel dealId=%s", did)
            return {"dealId": did, "dry_run": True}
        url = f"{self.base_url}/workingorders/otc/{did}"
        # API v1 and v2 both exist; try v2 then v1.
        last_exc: IgApiError | None = None
        for version in ("2", "1"):
            resp = self._client.delete(url, headers=self._headers(version=version))
            if not resp.is_error:
                payload = resp.json() if resp.content else {}
                return payload if isinstance(payload, dict) else {"raw": payload}
            last_exc = IgApiError(
                format_ig_http_error(resp, action="cancel_working_order", url=url)
            )
            code = ""
            try:
                code = str((resp.json() or {}).get("errorCode") or "")
            except Exception:
                pass
            if "not.found" not in code.lower():
                raise last_exc
        assert last_exc is not None
        raise last_exc

    def amend_order(self, order_id: str, *, level: float) -> WorkingOrder:
        order = self.ledger.amend_order(order_id, level=level)
        if self.dry_run or not self._cst:
            return order
        # IG amend requires dealId; keep ledger as source in V1 dry path
        logger.info("IG amend requested for %s -> %s", order_id, level)
        return order

    def cancel_order(self, order_id: str) -> None:
        order = self.ledger.working_orders.get(order_id)
        deal_id = (order.deal_id if order else "") or ""
        self.ledger.cancel_order(order_id)
        if self.dry_run or not self._cst:
            return
        if not deal_id:
            # Best-effort: match by client_ref via open working orders
            logger.warning("IG cancel skipped (no dealId) for local order %s", order_id)
            return
        self.cancel_working_order(deal_id)

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

    def _ig_working_order_body(
        self, order: WorkingOrder, *, currency: str | None = None
    ) -> dict[str, Any]:
        ccy = (currency or self.resolve_order_currency()).strip().upper()
        expiry = self.resolve_order_expiry()
        level = self.snap_level(float(order.level))
        order.level = level
        return {
            "epic": self.config.epic,
            "expiry": expiry,
            "direction": "BUY" if order.side == Side.BUY else "SELL",
            "size": float(order.size),
            "level": level,
            "type": "LIMIT" if order.type == OrderType.LIMIT else "STOP",
            "currencyCode": ccy,
            "timeInForce": "GOOD_TILL_CANCELLED",
            "guaranteedStop": False,
            # IG requires forceOpen=true for LIMIT working orders.
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
