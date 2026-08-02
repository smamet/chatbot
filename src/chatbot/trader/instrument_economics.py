"""Derive point_value and PnL currency from IG market details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chatbot.trader.point_size import infer_point_size
from chatbot.trader.profiles import get_profile, point_value_for_symbol


@dataclass(frozen=True)
class InstrumentEconomics:
    point_value: float
    currency: str
    point_size: float
    value_of_one_pip: float
    source: str  # ig | heuristic | profile


_CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "CHF": "CHF",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
}


def currency_symbol(code: str | None) -> str:
    key = str(code or "").strip().upper()
    if not key:
        return "$"
    return _CURRENCY_SYMBOLS.get(key, key)


def _mid_from_market(market: dict[str, Any] | None) -> float:
    if not isinstance(market, dict):
        return 0.0
    snap = market.get("snapshot") or {}
    if not isinstance(snap, dict):
        return 0.0
    try:
        bid = float(snap.get("bid") or 0)
        offer = float(snap.get("offer") or 0)
    except (TypeError, ValueError):
        return 0.0
    if bid > 0 and offer > 0:
        return (bid + offer) / 2.0
    return 0.0


def _point_size_from_market(market: dict[str, Any] | None, *, mid: float = 0.0) -> float:
    if isinstance(market, dict):
        snap = market.get("snapshot") or {}
        if isinstance(snap, dict):
            try:
                sf = float(snap.get("scalingFactor") or 0)
            except (TypeError, ValueError):
                sf = 0.0
            if sf > 1.0:
                return 1.0 / sf
    ref = mid or _mid_from_market(market)
    return infer_point_size(ref)


def _value_of_one_pip(market: dict[str, Any] | None) -> float | None:
    if not isinstance(market, dict):
        return None
    instrument = market.get("instrument") or {}
    if not isinstance(instrument, dict):
        return None
    raw = instrument.get("valueOfOnePip")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _currency_codes(market: dict[str, Any] | None) -> list[str]:
    if not isinstance(market, dict):
        return []
    instrument = market.get("instrument") or {}
    if not isinstance(instrument, dict):
        return []
    codes: list[str] = []
    for row in instrument.get("currencies") or []:
        if isinstance(row, dict) and row.get("code"):
            code = str(row["code"]).strip().upper()
            if code and code not in codes:
                codes.append(code)
    return codes


def pick_pnl_currency(
    *,
    market: dict[str, Any] | None = None,
    account_currency: str = "",
    fallback: str = "EUR",
) -> str:
    allowed = _currency_codes(market)
    acc = str(account_currency or "").strip().upper()
    if acc and (not allowed or acc in allowed):
        return acc
    for prefer in ("EUR", "USD", "GBP"):
        if prefer in allowed:
            return prefer
    if allowed:
        return allowed[0]
    return str(fallback or "EUR").strip().upper() or "EUR"


def resolve_instrument_economics(
    market: dict[str, Any] | None = None,
    *,
    account_currency: str = "",
    point_size: float | None = None,
    symbol: str | None = None,
    profile_id: str | None = None,
) -> InstrumentEconomics:
    """
    ``point_value = valueOfOnePip / point_size``.

    Falls back to profile/heuristic when IG fields are missing.
    """
    mid = _mid_from_market(market)
    ps = float(point_size) if point_size and point_size > 0 else _point_size_from_market(market, mid=mid)
    if ps <= 0:
        ps = 1.0

    pip_value = _value_of_one_pip(market)
    profile = get_profile(profile_id) if profile_id else None
    currency_fallback = "USD" if (mid and mid < 50) else "EUR"
    if profile is not None:
        # EURUSD profile → USD; indices → EUR
        currency_fallback = "USD" if profile.id == "eurusd" else "EUR"

    currency = pick_pnl_currency(
        market=market,
        account_currency=account_currency,
        fallback=currency_fallback,
    )

    if pip_value is not None:
        return InstrumentEconomics(
            point_value=float(pip_value) / ps,
            currency=currency,
            point_size=ps,
            value_of_one_pip=float(pip_value),
            source="ig",
        )

    # Heuristic: $1 (or €1) per IG point → point_value = 1 / point_size
    if market is not None or mid > 0:
        return InstrumentEconomics(
            point_value=1.0 / ps,
            currency=currency,
            point_size=ps,
            value_of_one_pip=1.0,
            source="heuristic",
        )

    pv = point_value_for_symbol(symbol, profile_id=profile_id)
    return InstrumentEconomics(
        point_value=float(pv),
        currency=currency,
        point_size=ps,
        value_of_one_pip=float(pv) * ps,
        source="profile",
    )


def resolve_economics_from_ig(
    ig: Any,
    *,
    epic: str | None = None,
    symbol: str | None = None,
    profile_id: str | None = None,
) -> InstrumentEconomics:
    """Fetch market via connector and resolve economics."""
    market: dict[str, Any] | None = None
    account_ccy = ""
    try:
        market = ig.get_market(epic) if epic else ig.get_market()
    except Exception:
        market = None
    try:
        acc = ig.get_account() if hasattr(ig, "get_account") else None
        if isinstance(acc, dict):
            account_ccy = str(acc.get("currency") or acc.get("preferredCurrency") or "")
    except Exception:
        account_ccy = ""
    point_size = None
    try:
        point_size = float(ig.resolve_point_size(epic=epic))
    except Exception:
        point_size = None
    return resolve_instrument_economics(
        market,
        account_currency=account_ccy,
        point_size=point_size,
        symbol=symbol,
        profile_id=profile_id,
    )
