"""Dump raw IG working-order request/response for DEMO diagnostics.

Usage (inside API container)::

    python -m chatbot.trader.ig_wo_raw_probe
    python -m chatbot.trader.ig_wo_raw_probe --allow-market-orders
    python -m chatbot.trader.ig_wo_raw_probe IX.D.SPTRD.IFA.IP

Or via Sail::

    ./sail exec api python -m chatbot.trader.ig_wo_raw_probe
    ./sail exec api python -m chatbot.trader.ig_wo_raw_probe --allow-market-orders
    ./sail exec api python -m chatbot.trader.ig_wo_raw_probe --tenant cac-trader

Credentials default to the ``eurusd-trader`` bot IG connector (override with ``--tenant``).

Market open/close is off by default (DEMO fees/spread). Pass ``--allow-market-orders``
(or ``--with-market-orders``) to also probe a tiny BUY market open + immediate close
on the first target epic (pass a single epic to choose which).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _load_ig_config(*, prefer_slug: str | None = None) -> dict[str, Any]:
    from sqlalchemy import text

    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.engine import create_db_engine, session_factory
    from chatbot.application.connector_service import ConnectorService
    from chatbot.config.settings import get_settings

    prefer = (prefer_slug or "eurusd-trader").strip().lower()
    Factory = session_factory(create_db_engine(get_settings()))
    with Factory() as session:
        svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        rows = session.execute(text("SELECT id, slug FROM tenants")).fetchall()
        by_slug = {str(r.slug).strip().lower(): r for r in rows}

        ordered: list[Any] = []
        if prefer and prefer in by_slug:
            ordered.append(by_slug[prefer])
        for soft in ("eurusd-trader", "eurusd"):
            row = by_slug.get(soft)
            if row is not None and row not in ordered:
                ordered.append(row)
        for row in rows:
            if row not in ordered:
                ordered.append(row)

        found: list[dict[str, Any]] = []
        for row in ordered:
            cfg = svc.get_ig_config(row.id)
            if not cfg or not str(cfg.get("api_key") or "").strip():
                continue
            out = dict(cfg)
            out["_tenant_slug"] = row.slug
            slug = str(row.slug).strip().lower()
            if prefer and slug == prefer:
                return out
            found.append(out)
            if "eurusd" in slug:
                return out
        if found:
            return found[0]
    raise SystemExit("No IG connector credentials found in DB.")


def _mid(market: dict[str, Any]) -> tuple[float, float, float, str]:
    snap = market.get("snapshot") or {}
    bid = float(snap.get("bid") or 0)
    offer = float(snap.get("offer") or snap.get("ask") or 0)
    mid = (bid + offer) / 2.0 if bid > 0 and offer > 0 else 0.0
    if mid <= 0 and bid > 0:
        mid = bid
    # Share CFDs often expose TRADEABLE with null bid/offer; use high/low.
    if mid <= 0:
        try:
            hi = float(snap.get("high") or 0)
            lo = float(snap.get("low") or 0)
        except (TypeError, ValueError):
            hi = lo = 0.0
        if hi > 0 and lo > 0:
            return lo, hi, (hi + lo) / 2.0, str(snap.get("marketStatus") or "—")
    return bid, offer, mid, str(snap.get("marketStatus") or "—")


def _min_dist(market: dict[str, Any], mid: float) -> float:
    rules = market.get("dealingRules") or {}
    row = rules.get("minNormalStopOrLimitDistance") or {}
    try:
        value = float(row.get("value") or 0)
    except (TypeError, ValueError):
        value = 0.0
    unit = str(row.get("unit") or "POINTS").upper()
    if value <= 0:
        return 12.0
    if unit == "PERCENTAGE" and mid > 0:
        return mid * value / 100.0
    return value


def _min_size(market: dict[str, Any]) -> float:
    rules = market.get("dealingRules") or {}
    row = rules.get("minDealSize") or {}
    try:
        value = float(row.get("value") or 0)
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0 else 1.0


def _currency(market: dict[str, Any], account_ccy: str) -> str:
    instrument = market.get("instrument") or {}
    codes = [
        str(r["code"]).strip().upper()
        for r in (instrument.get("currencies") or [])
        if isinstance(r, dict) and r.get("code")
    ]
    if account_ccy and account_ccy in codes:
        return account_ccy
    for prefer in ("EUR", "USD", "GBP"):
        if prefer in codes:
            return prefer
    return codes[0] if codes else (account_ccy or "EUR")


def _expiry(market: dict[str, Any]) -> str:
    instrument = market.get("instrument") or {}
    expiry = str(instrument.get("expiry") or "").strip()
    return expiry if expiry and expiry not in ("null", "None") else "-"


def _decimals(mid: float) -> int:
    if mid < 10:
        return 5
    if mid < 500:
        return 2
    return 1


def _offset(mid: float, min_d: float) -> float:
    # For FX / low-priced instruments, "points" minDist can exceed the
    # price itself — fall back to a % of mid so level stays positive.
    # Shares also need a % cushion when mid comes from high/low only.
    if mid < 50:
        return max(min(min_d, mid * 0.05) * 2.0, mid * 0.005)
    if mid < 1000:
        return max(min_d * 3.0, mid * 0.05, 2.0)
    return max(min_d * 2.0, 25.0)


def _tp_clearance(mid: float, min_d: float, offset: float) -> float:
    """Clearance for attached BUY TP above entry and above live offer.

    IG validates ``limitLevel`` against the live quote at place time.
    ``level + offset`` alone puts BUY LIMIT TP ≈ mid → ATTACHED_ORDER_LEVEL_ERROR.
    """
    if mid < 50:
        return max(offset, mid * 0.01)
    if mid < 1000:
        return max(offset, min_d * 2.0, mid * 0.02)
    return max(offset, min_d * 2.0, 25.0)


def _dump(title: str, payload: Any) -> None:
    print(title)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _post_wo_raw(ig: Any, body: dict[str, Any], *, version: str = "2") -> dict[str, Any]:
    """POST /workingorders/otc and return http + confirm payloads (no raise on reject)."""
    url = f"{ig.base_url}/workingorders/otc"
    headers = ig._headers(version=version)
    # Redact session tokens in printed headers; keep API key masked.
    printable_headers = {
        k: (
            "***"
            if k.upper() in {"CST", "X-SECURITY-TOKEN"}
            else (_mask(v) if k.upper() == "X-IG-API-KEY" else v)
        )
        for k, v in headers.items()
    }
    resp = ig._client.post(url, headers=headers, json=body)
    http_json: Any
    try:
        http_json = resp.json() if resp.content else None
    except Exception:
        http_json = resp.text
    out: dict[str, Any] = {
        "request": {
            "method": "POST",
            "url": url,
            "version": version,
            "headers": printable_headers,
            "body": body,
        },
        "http": {
            "status_code": resp.status_code,
            "body": http_json,
        },
        "confirm": None,
    }
    deal_ref = ""
    if isinstance(http_json, dict):
        deal_ref = str(http_json.get("dealReference") or "").strip()
    if deal_ref and not resp.is_error:
        confirm_url = f"{ig.base_url}/confirms/{deal_ref}"
        conf_resp = ig._client.get(confirm_url, headers=ig._headers(version="1"))
        try:
            conf_json = conf_resp.json() if conf_resp.content else None
        except Exception:
            conf_json = conf_resp.text
        out["confirm"] = {
            "method": "GET",
            "url": confirm_url,
            "status_code": conf_resp.status_code,
            "body": conf_json,
        }
        did = ""
        if isinstance(conf_json, dict):
            did = str(conf_json.get("dealId") or "").strip()
            if str(conf_json.get("dealStatus") or "").upper() == "ACCEPTED" and did:
                try:
                    ig.cancel_working_order(did)
                    out["cancelled_deal_id"] = did
                except Exception as exc:
                    out["cancel_error"] = str(exc)
    return out


def _mask(value: str, *, keep: int = 4) -> str:
    raw = str(value or "")
    if len(raw) <= keep * 2:
        return "*" * len(raw)
    return f"{raw[:keep]}…{raw[-keep:]}"


def _attach_rule(market: dict[str, Any]) -> tuple[float, str, float, str]:
    """Return (min_value, min_unit, max_value, max_unit) from dealingRules."""
    rules = market.get("dealingRules") or {}
    mn = rules.get("minNormalStopOrLimitDistance") or {}
    mx = rules.get("maxStopOrLimitDistance") or {}
    try:
        min_v = float(mn.get("value") or 0)
    except (TypeError, ValueError):
        min_v = 0.0
    try:
        max_v = float(mx.get("value") or 0)
    except (TypeError, ValueError):
        max_v = 0.0
    min_u = str(mn.get("unit") or "POINTS").upper()
    max_u = str(mx.get("unit") or "POINTS").upper()
    return min_v, min_u, max_v, max_u


def _build_body(
    *,
    epic: str,
    market: dict[str, Any],
    account_ccy: str,
    order_type: str,
    with_tp: bool,
) -> dict[str, Any] | None:
    bid, offer, mid, status = _mid(market)
    if mid <= 0 or status != "TRADEABLE":
        return None
    min_d = _min_dist(market, mid)
    size = _min_size(market)
    ccy = _currency(market, account_ccy)
    expiry = _expiry(market)
    offset = _offset(mid, min_d)
    decimals = _decimals(mid)
    order_type = order_type.upper()
    level = round(mid - offset, decimals) if order_type == "LIMIT" else round(mid + offset, decimals)
    if level <= 0:
        return None
    body: dict[str, Any] = {
        "epic": epic,
        "expiry": expiry,
        "direction": "BUY",
        "size": size,
        "level": level,
        "type": order_type,
        "currencyCode": ccy,
        "timeInForce": "GOOD_TILL_CANCELLED",
        "guaranteedStop": False,
        "forceOpen": True,
    }
    if with_tp:
        min_v, _min_u, max_v, max_u = _attach_rule(market)
        # EURUSD CFD: maxStopOrLimitDistance is 75 POINTS (not %). Absolute
        # limitLevel far from entry converts to >>75 pts → ATTACHED_ORDER_LEVEL_ERROR.
        # Attach via limitDistance within [min, max] instead.
        if mid < 50 and max_u == "POINTS" and max_v > 0:
            lo = max(min_v * 2.0 if min_v > 0 else 10.0, 10.0)
            hi = max_v * 0.8 if max_v > lo else max_v
            dist = min(max(lo, 20.0), hi) if hi > 0 else 20.0
            body["limitDistance"] = float(round(dist, 2))
        else:
            clear = _tp_clearance(mid, min_d, offset)
            live_ref = offer if offer > 0 else mid
            body["limitLevel"] = round(max(level + clear, live_ref + clear), decimals)
    # Echo market context outside IG body (printed alongside).
    body["_probe_meta"] = {
        "market_epic": epic,
        "instrument_name": str((market.get("instrument") or {}).get("name") or ""),
        "market_status": status,
        "bid": bid,
        "offer": offer,
        "mid": mid,
        "minNormalStopOrLimitDistance": min_d,
        "shape": f"BUY {order_type}{' +TP' if with_tp else ''}",
    }
    return body


def _resolve_tradeable_epic(
    ig: Any,
    *,
    search_term: str,
    fallbacks: tuple[str, ...],
    prefer_substrings: tuple[str, ...] = (),
) -> tuple[str, str] | None:
    """Pick a TRADEABLE epic via search, else first fallback that GET /markets accepts."""
    try:
        rows = ig.search_markets(search_term)
    except Exception:
        rows = []
    ranked: list[tuple[int, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        epic = str(row.get("epic") or "").strip()
        status = str(row.get("marketStatus") or "").upper()
        if not epic or (status and status != "TRADEABLE"):
            continue
        name = str(row.get("instrumentName") or row.get("instrument") or epic)
        upper = f"{epic} {name}".upper()
        # Skip knock-outs / weekend-only noise for defaults.
        if any(x in upper for x in ("BULL", "BEAR", "WEEKEND", " KO")):
            continue
        score = 50
        if "ITA" in upper or epic.upper().startswith("EF.D."):
            score += 40
        if "CASH" in epic.upper():
            score -= 15
        for i, pref in enumerate(prefer_substrings):
            if pref.upper() in upper:
                score -= 25 - i
        ranked.append((score, epic, name))
    if ranked:
        ranked.sort()
        _score, epic, name = ranked[0]
        return epic, name
    for epic in fallbacks:
        try:
            if hasattr(ig, "_market_cache"):
                ig._market_cache.pop(epic, None)
            market = ig.get_market(epic)
        except Exception:
            continue
        snap = (market.get("snapshot") or {}) if isinstance(market, dict) else {}
        if str(snap.get("marketStatus") or "").upper() == "TRADEABLE":
            name = str(
                ((market.get("instrument") or {}) if isinstance(market, dict) else {}).get("name")
                or epic
            )
            return epic, name
    return None


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IG DEMO raw working-order probe (LIMIT/STOP ± TP)."
    )
    parser.add_argument(
        "--tenant",
        "--slug",
        dest="tenant_slug",
        default="eurusd-trader",
        help="Tenant slug whose IG connector credentials to use (default: eurusd-trader).",
    )
    parser.add_argument(
        "--allow-market-orders",
        "--with-market-orders",
        action="store_true",
        dest="allow_market_orders",
        help=(
            "Also probe market open + immediate close on the first target epic "
            "(DEMO fees/spread)."
        ),
    )
    parser.add_argument(
        "epics",
        nargs="*",
        help="Optional epic overrides (skip default multi-market search).",
    )
    return parser.parse_args(argv)


def _probe_market_open_close(
    ig: Any,
    *,
    epic: str,
    instrument: str,
    account_ccy: str,
    record: Any,
) -> None:
    """Tiny DEMO market BUY + close; record summary row."""
    from chatbot.trader.models import LegRole, Side

    shape = "BUY MARKET open/close"
    print("\n" + "-" * 72)
    print(f"PROBE market={epic} shape={shape} instrument={instrument!r}")
    print("(allow_market_orders=true — DEMO fees/spread may apply)")
    try:
        size = max(float(ig.resolve_min_deal_size(epic=epic) or 1.0), 0.1)
        ccy = ig.resolve_order_currency(epic=epic) or account_ccy or "EUR"
        prev = ig.config.epic
        ig.config.epic = epic
        try:
            leg_id = ig.open_market_position(
                Side.BUY, size, role=LegRole.HEDGE, currency=ccy
            )
            print(f"MARKET OPEN BUY size={size} ccy={ccy} leg={leg_id}")
            ig.market_close(leg_id)
            print(f"MARKET CLOSE leg={leg_id}")
            record(
                epic=epic,
                instrument=instrument,
                shape=shape,
                outcome="ACCEPTED",
                deal_id=str(leg_id or ""),
            )
        finally:
            ig.config.epic = prev
    except Exception as exc:
        print(f"MARKET probe FAILED: {exc}")
        record(
            epic=epic,
            instrument=instrument,
            shape=shape,
            outcome="ERROR",
            reason=str(exc),
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(list(argv or sys.argv[1:]))
    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgConnector

    cfg = _load_ig_config(prefer_slug=str(args.tenant_slug or "eurusd-trader"))
    configured = str(cfg.get("epic") or "IX.D.CAC.BMU.IP").strip()
    explicit_epics = [ep.strip() for ep in args.epics if ep and str(ep).strip()]
    allow_market_orders = bool(args.allow_market_orders)

    cac = TraderConfig(
        ig_api_key=str(cfg.get("api_key") or ""),
        ig_username=str(cfg.get("username") or ""),
        ig_password=str(cfg.get("password") or ""),
        ig_account_id=str(cfg.get("account_id") or ""),
        ig_acc_type="DEMO",
        epic=configured,
        order_size=1.0,
        allow_market_orders=allow_market_orders,
    )
    ig = IgConnector(cac, dry_run=False)
    try:
        ig.login()
        account = ig.get_active_account()
        account_ccy = str(
            account.get("currency") or account.get("preferredCurrency") or ""
        ).strip().upper()

        if explicit_epics:
            targets = [(ep, ep) for ep in explicit_epics]
        else:
            targets = [(configured, "configured (France 40)")]
            resolved_list: list[tuple[str, str] | None] = [
                _resolve_tradeable_epic(
                    ig,
                    search_term="Germany 40",
                    fallbacks=("IX.D.DAX.IFMM.IP", "IX.D.DAX.IFD.IP", "IX.D.DAX.BMU.IP"),
                    prefer_substrings=("GERMANY 40", "DAX", "CASH"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="US 500",
                    fallbacks=("IX.D.SPTRD.IFA.IP", "IX.D.SPTRD.DAILY.IP"),
                    prefer_substrings=("US 500", "SPTRD", "CASH"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="US Tech 100",
                    fallbacks=("IX.D.NASDAQ.IFA.IP", "IX.D.NASDAQ.DAILY.IP"),
                    prefer_substrings=("US TECH 100", "NASDAQ", "CASH"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="EUR/USD",
                    fallbacks=("CS.D.EURUSD.MINI.IP", "CS.D.EURUSD.CFD.IP"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="Spot Gold",
                    fallbacks=("CS.D.IN_GOLD.MFI.IP", "CS.D.CFDGOLD.BMU.IP"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="Oil - Brent",
                    fallbacks=("CC.D.LCO.UME.IP", "CC.D.LCO.IPC.IP"),
                    prefer_substrings=("BRENT", "LCO", "CRUDE"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="Liquide",
                    fallbacks=("EC.D.AI.CASH.IP",),
                    prefer_substrings=("AIR LIQUIDE", "L'AIR LIQUIDE", "CASH"),
                ),
                _resolve_tradeable_epic(
                    ig,
                    search_term="NVIDIA",
                    fallbacks=("UC.D.NVDA.CASH.IP", "UA.D.NVDA.CASH.IP"),
                    prefer_substrings=("NVIDIA", "NVDA", "CASH"),
                ),
            ]
            labels = (
                "DAX / Germany 40",
                "US500",
                "NASDAQ / US Tech 100",
                "EUR/USD",
                "Gold",
                "Brent",
                "Air Liquide",
                "NVIDIA",
            )
            for resolved, fallback_label in zip(resolved_list, labels, strict=True):
                if resolved:
                    targets.append(resolved)
                else:
                    print(f"WARN: could not resolve TRADEABLE epic for {fallback_label}")

        print("=" * 72)
        print("IG DEMO raw working-order probe")
        print(f"tenant={cfg.get('_tenant_slug')} account_id={cfg.get('account_id')}")
        print(f"accountType={account.get('accountType')} currency={account_ccy}")
        print(f"host={ig.base_url}")
        print(f"allow_market_orders={str(allow_market_orders).lower()}")
        print("targets:")
        for epic, label in targets:
            print(f"  - {epic}  ({label})")
        print("=" * 72)

        shapes = (
            ("LIMIT", False),
            ("LIMIT", True),
            ("STOP", False),
            ("STOP", True),
        )
        summary_rows: list[dict[str, str]] = []

        def _record(
            *,
            epic: str,
            instrument: str,
            shape: str,
            outcome: str,
            reason: str = "",
            deal_ref: str = "",
            deal_id: str = "",
        ) -> None:
            summary_rows.append(
                {
                    "epic": epic,
                    "instrument": instrument,
                    "shape": shape,
                    "outcome": outcome,
                    "reason": reason,
                    "dealReference": deal_ref,
                    "dealId": deal_id,
                }
            )

        for epic, label in targets:
            print("\n" + "#" * 72)
            print(f"# MARKET epic={epic}  ({label})")
            print("#" * 72)
            try:
                if hasattr(ig, "_market_cache"):
                    ig._market_cache.pop(epic, None)
                ig.config.epic = epic
                market = ig.get_market(epic)
            except Exception as exc:
                print(f"ERROR get_market({epic}): {exc}")
                for order_type, with_tp in shapes:
                    _record(
                        epic=epic,
                        instrument="",
                        shape=f"BUY {order_type}{' +TP' if with_tp else ''}",
                        outcome="ERROR",
                        reason=str(exc),
                    )
                continue
            bid, offer, mid, status = _mid(market)
            name = str((market.get("instrument") or {}).get("name") or "")
            print(
                f"instrument={name!r} status={status} bid={bid} offer={offer} mid={mid}"
            )
            for order_type, with_tp in shapes:
                shape = f"BUY {order_type}{' +TP' if with_tp else ''}"
                built = _build_body(
                    epic=epic,
                    market=market,
                    account_ccy=account_ccy,
                    order_type=order_type,
                    with_tp=with_tp,
                )
                print("\n" + "-" * 72)
                if built is None:
                    print(f"SKIP epic={epic} {shape} (not TRADEABLE or no quote)")
                    _record(
                        epic=epic,
                        instrument=name,
                        shape=shape,
                        outcome="SKIP",
                        reason=f"status={status} mid={mid}",
                    )
                    continue
                meta = built.pop("_probe_meta")
                print(
                    f"PROBE market={meta['market_epic']} shape={meta['shape']} "
                    f"instrument={meta['instrument_name']!r}"
                )
                _dump("REQUEST body (POST /workingorders/otc):", built)
                print(f"market_epic={meta['market_epic']}")
                result = _post_wo_raw(ig, built)
                result["market_epic"] = epic
                result["instrument_name"] = meta["instrument_name"]
                result["shape"] = meta["shape"]
                _dump("RAW exchange:", result)

                conf_body = (result.get("confirm") or {}).get("body")
                if not isinstance(conf_body, dict):
                    http_body = result.get("http") or {}
                    _record(
                        epic=epic,
                        instrument=name,
                        shape=shape,
                        outcome="ERROR",
                        reason=f"HTTP {http_body.get('status_code')} no confirm",
                    )
                    continue
                deal_status = str(conf_body.get("dealStatus") or "").upper() or "—"
                reason = str(conf_body.get("reason") or "")
                _record(
                    epic=epic,
                    instrument=name,
                    shape=shape,
                    outcome=deal_status,
                    reason=reason,
                    deal_ref=str(conf_body.get("dealReference") or ""),
                    deal_id=str(conf_body.get("dealId") or ""),
                )

        # Market open/close once on the first target (avoids DEMO fee burn across
        # the full multi-market matrix). Pass a single epic + the flag to target
        # another instrument.
        if allow_market_orders:
            epic, label = targets[0]
            print("\n" + "#" * 72)
            print(f"# MARKET open/close epic={epic}  ({label})")
            print("#" * 72)
            try:
                if hasattr(ig, "_market_cache"):
                    ig._market_cache.pop(epic, None)
                ig.config.epic = epic
                market = ig.get_market(epic)
                bid, offer, mid, status = _mid(market)
                name = str((market.get("instrument") or {}).get("name") or "")
            except Exception as exc:
                print(f"ERROR get_market({epic}): {exc}")
                _record(
                    epic=epic,
                    instrument="",
                    shape="BUY MARKET open/close",
                    outcome="ERROR",
                    reason=str(exc),
                )
            else:
                if status != "TRADEABLE" or mid <= 0:
                    print(
                        f"SKIP epic={epic} BUY MARKET open/close "
                        f"(status={status} mid={mid})"
                    )
                    _record(
                        epic=epic,
                        instrument=name,
                        shape="BUY MARKET open/close",
                        outcome="SKIP",
                        reason=f"status={status} mid={mid}",
                    )
                else:
                    _probe_market_open_close(
                        ig,
                        epic=epic,
                        instrument=name,
                        account_ccy=account_ccy,
                        record=_record,
                    )
        else:
            print("\nMarket open/close skipped (allow_market_orders=false).")

        _print_summary(summary_rows)
        print("\nDONE")
        return 0
    finally:
        ig.close()


def _print_summary(rows: list[dict[str, str]]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if not rows:
        print("(no probes)")
        return

    # Compact table
    col_epic = max(len("epic"), max(len(r["epic"]) for r in rows))
    col_shape = max(len("shape"), max(len(r["shape"]) for r in rows))
    col_out = max(len("outcome"), max(len(r["outcome"]) for r in rows))
    header = (
        f"{'epic'.ljust(col_epic)}  {'shape'.ljust(col_shape)}  "
        f"{'outcome'.ljust(col_out)}  reason"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        reason = r["reason"] or "—"
        print(
            f"{r['epic'].ljust(col_epic)}  {r['shape'].ljust(col_shape)}  "
            f"{r['outcome'].ljust(col_out)}  {reason}"
        )

    accepted = sum(1 for r in rows if r["outcome"] == "ACCEPTED")
    rejected = sum(1 for r in rows if r["outcome"] == "REJECTED")
    skipped = sum(1 for r in rows if r["outcome"] == "SKIP")
    errors = sum(1 for r in rows if r["outcome"] == "ERROR")
    print("-" * len(header))
    print(
        f"totals: ACCEPTED={accepted} REJECTED={rejected} "
        f"SKIP={skipped} ERROR={errors} (n={len(rows)})"
    )

    # Per-market rollup
    by_epic: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_epic.setdefault(r["epic"], []).append(r)
    print("\nPer market:")
    for epic, group in by_epic.items():
        inst = next((g["instrument"] for g in group if g["instrument"]), "")
        acc = sum(1 for g in group if g["outcome"] == "ACCEPTED")
        rej = sum(1 for g in group if g["outcome"] == "REJECTED")
        sk = sum(1 for g in group if g["outcome"] == "SKIP")
        err = sum(1 for g in group if g["outcome"] == "ERROR")
        reasons = sorted(
            {
                g["reason"]
                for g in group
                if g["outcome"] == "REJECTED" and g["reason"]
            }
        )
        label = f"{epic}" + (f" ({inst})" if inst else "")
        line = f"  {label}: ACCEPTED={acc} REJECTED={rej} SKIP={sk} ERROR={err}"
        if reasons:
            line += f"  [{', '.join(reasons)}]"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
