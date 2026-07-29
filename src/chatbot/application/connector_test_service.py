from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from chatbot.adapters.mail.factory import build_email_sender
from chatbot.adapters.mail.imap_client import ImapError, ImapMailClient
from chatbot.adapters.mail.smtp_sender import EmailSendError, SmtpEmailSender, _parse_use_tls
from chatbot.application.mail_connection_service import MailConnectionService
from chatbot.application.mail_oauth_service import MailOAuthError, prepare_oauth_mail_config
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.models.connector_schema import (
    EmailAuthType,
    EmailOutboundProvider,
    is_oauth_auth_type,
    resolve_email_auth_type,
    resolve_email_outbound_provider,
)
from chatbot.domain.models.mail_connection import MailConnection


@dataclass(frozen=True, slots=True)
class ConnectorTestResult:
    ok: bool
    message: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_connector_connection_test(
    connector_type: str,
    direction: str,
    config: dict,
    *,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    try:
        if connector_type == "ig":
            return _test_ig(config)
        if connector_type == "email":
            if direction == "in":
                return _test_imap(config, session=session, tenant_id=tenant_id, settings=settings)
            if direction == "out":
                return _test_outbound(config, session=session, tenant_id=tenant_id, settings=settings)
            return ConnectorTestResult(
                ok=False, message="Invalid connector direction.", error="invalid_direction"
            )
        return ConnectorTestResult(
            ok=False,
            message="Connection test is only available for email and IG connectors.",
            error="unsupported_connector",
        )
    except (ImapError, EmailSendError, MailOAuthError) as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))
    except Exception as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))


def _mask_secret(value: str, *, keep: int = 4) -> str:
    raw = (value or "").strip()
    if not raw:
        return "(empty)"
    if len(raw) <= keep * 2:
        return "*" * len(raw)
    return f"{raw[:keep]}…{raw[-keep:]} ({len(raw)} chars)"


def _test_ig(config: dict) -> ConnectorTestResult:
    """Login to IG and fetch a couple of 15m bars for the configured epic."""
    import httpx

    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgAuthError, IgConnector, _IG_HOSTS, format_ig_http_error

    api_key = str(config.get("api_key", "")).strip()
    username = str(config.get("username", "")).strip()
    password = str(config.get("password", "")).strip()
    if not api_key or not username or not password:
        return ConnectorTestResult(
            ok=False,
            message="IG API key, username, and password are required (save first, or fill them).",
            error="missing_credentials",
        )
    acc_type = str(config.get("acc_type", "DEMO") or "DEMO").strip().upper()
    if acc_type not in ("DEMO", "LIVE"):
        acc_type = "DEMO"
    epic = str(config.get("epic", "") or "IX.D.CAC.BMU.IP").strip()
    account_id = str(config.get("account_id", "")).strip()
    base_url = _IG_HOSTS.get(acc_type, _IG_HOSTS["DEMO"])
    context = (
        f"env={acc_type}\n"
        f"host={base_url}\n"
        f"username={username}\n"
        f"api_key={_mask_secret(api_key)}\n"
        f"password={'(set)' if password else '(empty)'}\n"
        f"account_id={account_id or '(none)'}\n"
        f"epic={epic}"
    )
    cfg = TraderConfig(
        ig_api_key=api_key,
        ig_username=username,
        ig_password=password,
        ig_account_id=account_id,
        ig_acc_type=acc_type,
        epic=epic,
    )
    ig = IgConnector(cfg, dry_run=True)
    logged_in = False
    try:
        ig.login()
        logged_in = True
        if not ig._cst or not ig._security:
            return ConnectorTestResult(
                ok=False,
                message="IG login failed (no session tokens returned).",
                error=context,
            )
        df = ig.get_ohlc("15m", 2)
        if df.empty:
            return ConnectorTestResult(
                ok=True,
                message=(
                    f"IG {acc_type} login OK for {epic}, but no 15m prices returned.\n\n{context}"
                ),
            )
        last = float(df["close"].iloc[-1])
        ts = df.index[-1]
        return ConnectorTestResult(
            ok=True,
            message=(
                f"IG {acc_type} login OK · {epic} last 15m close {last:.2f} @ {ts}\n\n{context}"
            ),
        )
    except IgAuthError as exc:
        prefix = "IG login OK, but next step failed.\n" if logged_in else ""
        return ConnectorTestResult(ok=False, message=f"{prefix}{exc}", error=context)
    except httpx.HTTPStatusError as exc:
        detail = format_ig_http_error(exc.response, action="request", url=str(exc.request.url))
        return ConnectorTestResult(ok=False, message=detail, error=context)
    except httpx.RequestError as exc:
        return ConnectorTestResult(
            ok=False,
            message=f"IG network error: {exc}",
            error=context,
        )
    finally:
        ig.close()


def _diagnose_ig_working_order_rejects(
    ig: Any,
    *,
    mid: float,
    size: float,
    currency: str,
    expiry: str,
    epic: str,
    min_dist: float,
    lines: list[str],
) -> dict[str, Any]:
    """When bare BUY LIMIT fails at every offset, probe shape/payload/epic.

    Returns ``{"accepted_deal_id": ...}`` if any probe was ACCEPTED (caller cancels).
    """
    from chatbot.trader.ig_connector import IgApiError

    off = max(float(min_dist) * 2.0, 25.0)
    if mid < 50:
        off = max(float(min_dist) * 2.0, mid * 0.005)
        off = min(off, mid * 0.1) if mid > 0 else off
    bid, offer = (0.0, 0.0)
    try:
        bid, offer = ig.dealable_quote()
    except Exception:
        pass
    lines.append("")
    lines.append(
        f"DIAG quote bid={bid} offer={offer} mid={mid:.5f} probe_offset={off:.5f}"
    )
    # Existing positions with attached stop/limit are the prime suspect: IG
    # rejects new resting orders on the instrument with ATTACHED_ORDER_LEVEL_ERROR
    # while opposing exposure carries attached orders.
    attached_on_positions = 0
    try:
        positions = ig.list_open_positions(epic="")
        n_wo = len(ig.list_working_orders())
        lines.append(
            f"DIAG account book: open_positions={len(positions)} open_working_orders={n_wo}"
        )
        for pos in positions:
            has_attach = pos.get("stop_level") is not None or pos.get("limit_level") is not None
            if has_attach:
                attached_on_positions += 1
            lines.append(
                f"  · pos {pos.get('side').value if hasattr(pos.get('side'), 'value') else pos.get('side')} "
                f"size={pos.get('size')} @ {pos.get('level')} "
                f"stop={pos.get('stop_level')} limit={pos.get('limit_level')} "
                f"epic={pos.get('epic')} dealId={pos.get('deal_id')}"
            )
        if attached_on_positions:
            lines.append(
                f"  ⚠ {attached_on_positions} position(s) carry attached stop/limit — "
                "IG can reject ALL new working orders on this instrument with "
                "ATTACHED_ORDER_LEVEL_ERROR while these exist. Close the positions "
                "(or remove their attached orders) and re-run this test."
            )
    except Exception as exc:
        lines.append(f"DIAG account book: unavailable ({exc})")

    def _submit(label: str, body: dict[str, Any], *, version: str = "2") -> str:
        """Submit one probe, log the outcome, return dealId when ACCEPTED."""
        try:
            conf = ig.submit_working_order_raw(body, version=version)
            status = str(conf.get("dealStatus") or "").upper()
            reason = str(conf.get("reason") or "")
            did = str(conf.get("dealId") or "")
            if status == "ACCEPTED":
                lines.append(f"  ACCEPTED {label} dealId={did or '—'}")
                return did
            lines.append(f"  REJECTED {label} reason={reason or '—'} dealStatus={status or '—'}")
        except IgApiError as exc:
            lines.append(f"  ERROR {label}: {exc}")
        except Exception as exc:
            lines.append(f"  ERROR {label}: {type(exc).__name__}: {exc}")
        return ""

    def _base(*, direction: str, level: float, otype: str, ep: str = epic) -> dict[str, Any]:
        return {
            "epic": ep,
            "expiry": expiry,
            "direction": direction,
            "size": float(size),
            "level": float(level),
            "type": otype,
            "currencyCode": currency,
            "timeInForce": "GOOD_TILL_CANCELLED",
            "guaranteedStop": False,
            "forceOpen": True,
        }

    accepted_deal_id = ""

    def _keep_first_accepted(did: str) -> None:
        nonlocal accepted_deal_id
        if not did:
            return
        if not accepted_deal_id:
            accepted_deal_id = did
        elif did != accepted_deal_id:
            try:
                ig.cancel_working_order(did)
            except Exception:
                pass

    # Limited-risk accounts (e.g. IG France) require a guaranteed stop on every
    # opening order; a missing/invalid one can surface as ATTACHED_ORDER_LEVEL_ERROR.
    cr_points = 0.0
    try:
        mkt = ig.get_market()
        rules = (mkt.get("dealingRules") or {}) if isinstance(mkt, dict) else {}
        row = rules.get("minControlledRiskStopDistance") or {}
        val = float(row.get("value") or 0)
        unit = str(row.get("unit") or "POINTS").strip().upper()
        cr_points = mid * val / 100.0 if unit == "PERCENTAGE" else val
        spacing = float((rules.get("controlledRiskSpacing") or {}).get("value") or 0)
        cr_points += spacing + 5.0
    except Exception:
        pass
    if cr_points <= 0:
        cr_points = max(mid * 0.025, 50.0)
    cr_points = float(ig.snap_level(cr_points))
    buy_level = float(ig.snap_level(mid - off))

    lines.append("DIAG guaranteed-stop probes (limited-risk account?):")
    gs_full = _base(direction="BUY", level=buy_level, otype="LIMIT")
    gs_full["guaranteedStop"] = True
    gs_full["stopDistance"] = cr_points
    _keep_first_accepted(
        _submit(f"BUY LIMIT guaranteedStop=true stopDistance={cr_points}", gs_full)
    )
    gs_small = _base(direction="BUY", level=buy_level, otype="LIMIT")
    gs_small["guaranteedStop"] = True
    gs_small["stopDistance"] = max(float(min_dist), 15.0)
    _keep_first_accepted(
        _submit(
            f"BUY LIMIT guaranteedStop=true stopDistance={gs_small['stopDistance']}",
            gs_small,
        )
    )
    gs_lvl = _base(direction="BUY", level=buy_level, otype="LIMIT")
    gs_lvl["guaranteedStop"] = True
    gs_lvl["stopLevel"] = float(ig.snap_level(buy_level - cr_points))
    _keep_first_accepted(
        _submit(f"BUY LIMIT guaranteedStop=true stopLevel={gs_lvl['stopLevel']}", gs_lvl)
    )

    shapes: list[tuple[str, dict[str, Any]]] = [
        ("BUY LIMIT below", _base(direction="BUY", level=float(ig.snap_level(mid - off)), otype="LIMIT")),
        ("SELL LIMIT above", _base(direction="SELL", level=float(ig.snap_level(mid + off)), otype="LIMIT")),
        ("BUY STOP above", _base(direction="BUY", level=float(ig.snap_level(mid + off)), otype="STOP")),
        ("SELL STOP below", _base(direction="SELL", level=float(ig.snap_level(mid - off)), otype="STOP")),
    ]

    lines.append("DIAG shapes (current payload):")
    for label, body in shapes:
        try:
            conf = ig.submit_working_order_raw(body)
            status = str(conf.get("dealStatus") or "").upper()
            reason = str(conf.get("reason") or "")
            did = str(conf.get("dealId") or "")
            if status == "ACCEPTED":
                lines.append(
                    f"  ACCEPTED {label} @ {body['level']} dealId={did or '—'}"
                )
                if not accepted_deal_id and did:
                    accepted_deal_id = did
                elif did and did != accepted_deal_id:
                    try:
                        ig.cancel_working_order(did)
                    except Exception:
                        pass
            else:
                lines.append(
                    f"  REJECTED {label} @ {body['level']} "
                    f"reason={reason or '—'} dealStatus={status or '—'}"
                )
        except IgApiError as exc:
            lines.append(f"  ERROR {label}: {exc}")
        except Exception as exc:
            lines.append(f"  ERROR {label}: {type(exc).__name__}: {exc}")

    # Payload variants on BUY LIMIT — string booleans (trading-ig style), omit
    # fields, stopDistance attach, API v1, int size.
    stop_d = max(float(min_dist), 12.0)
    limit_d = max(float(min_dist), 12.0)
    variants: list[tuple[str, dict[str, Any], str]] = []
    base_buy = _base(direction="BUY", level=buy_level, otype="LIMIT")

    v_str = dict(base_buy)
    v_str["forceOpen"] = "true"
    v_str["guaranteedStop"] = "false"
    variants.append(("string_bools", v_str, "2"))

    v_omit_gs = dict(base_buy)
    v_omit_gs.pop("guaranteedStop", None)
    variants.append(("omit_guaranteedStop", v_omit_gs, "2"))

    v_omit_fo = dict(base_buy)
    v_omit_fo.pop("forceOpen", None)
    variants.append(("omit_forceOpen", v_omit_fo, "2"))

    v_stop_d = dict(base_buy)
    v_stop_d["stopDistance"] = stop_d
    variants.append((f"stopDistance={stop_d}", v_stop_d, "2"))

    v_lim_d = dict(base_buy)
    v_lim_d["limitDistance"] = limit_d
    variants.append((f"limitDistance={limit_d}", v_lim_d, "2"))

    v_both_d = dict(base_buy)
    v_both_d["stopDistance"] = stop_d
    v_both_d["limitDistance"] = limit_d
    variants.append((f"stop+limitDistance={stop_d}/{limit_d}", v_both_d, "2"))

    v_int = dict(base_buy)
    v_int["size"] = int(size) if float(size) == int(size) else size
    v_int["level"] = int(buy_level) if float(buy_level) == int(buy_level) else buy_level
    variants.append(("int_size_level", v_int, "2"))

    v_v1 = dict(base_buy)
    variants.append(("api_v1", v_v1, "1"))

    lines.append("DIAG payload variants (BUY LIMIT):")
    for label, body, ver in variants:
        try:
            conf = ig.submit_working_order_raw(body, version=ver)
            status = str(conf.get("dealStatus") or "").upper()
            reason = str(conf.get("reason") or "")
            did = str(conf.get("dealId") or "")
            if status == "ACCEPTED":
                lines.append(f"  ACCEPTED {label} dealId={did or '—'}")
                if not accepted_deal_id and did:
                    accepted_deal_id = did
                elif did and did != accepted_deal_id:
                    try:
                        ig.cancel_working_order(did)
                    except Exception:
                        pass
            else:
                lines.append(
                    f"  REJECTED {label} reason={reason or '—'} dealStatus={status or '—'}"
                )
        except IgApiError as exc:
            lines.append(f"  ERROR {label}: {exc}")
        except Exception as exc:
            lines.append(f"  ERROR {label}: {type(exc).__name__}: {exc}")

    # Alternate France 40 epics (one BUY LIMIT each).
    lines.append("DIAG alternate France 40 epics (BUY LIMIT):")
    epic_candidates: list[str] = []
    try:
        _chosen, seen = ig.find_compatible_epic(account_type=ig.resolve_account_type())
        for ep in seen:
            if ep and ep not in epic_candidates:
                epic_candidates.append(ep)
    except Exception as exc:
        lines.append(f"  epic search failed: {exc}")
    for hint in (
        "IX.D.CAC.IFD.IP",
        "IX.D.CAC.IFS.IP",
        "IX.D.CAC.CFS.IP",
        "IX.D.CAC.DAILY.IP",
        "IX.D.CAC.IMF.IP",
    ):
        if hint not in epic_candidates:
            epic_candidates.append(hint)
    # Prefer configured epic first (already failed) then others — skip duplicate work.
    tried = {epic}
    for alt in epic_candidates:
        if alt in tried:
            continue
        tried.add(alt)
        if len(tried) > 8:
            break
        try:
            # Refresh market for alternate epic so expiry/currency match.
            prev = ig.config.epic
            ig.config.epic = alt
            if hasattr(ig, "_market_cache"):
                ig._market_cache.pop(alt, None)
            mkt = ig.get_market(alt)
            snap = (mkt.get("snapshot") or {}) if isinstance(mkt, dict) else {}
            inst = (mkt.get("instrument") or {}) if isinstance(mkt, dict) else {}
            alt_expiry = str(inst.get("expiry") or expiry or "-").strip() or "-"
            alt_mid = mid
            try:
                bid_a = float(snap.get("bid") or 0)
                offer_a = float(snap.get("offer") or snap.get("ask") or 0)
                if bid_a > 0 and offer_a > 0:
                    alt_mid = (bid_a + offer_a) / 2.0
            except (TypeError, ValueError):
                pass
            codes = []
            for row in inst.get("currencies") or []:
                if isinstance(row, dict) and row.get("code"):
                    codes.append(str(row["code"]).strip().upper())
            alt_ccy = currency if currency in codes or not codes else codes[0]
            body = {
                "epic": alt,
                "expiry": alt_expiry,
                "direction": "BUY",
                "size": float(size),
                "level": float(ig.snap_level(alt_mid - off)),
                "type": "LIMIT",
                "currencyCode": alt_ccy,
                "timeInForce": "GOOD_TILL_CANCELLED",
                "guaranteedStop": False,
                "forceOpen": True,
            }
            conf = ig.submit_working_order_raw(body)
            status = str(conf.get("dealStatus") or "").upper()
            reason = str(conf.get("reason") or "")
            did = str(conf.get("dealId") or "")
            name = inst.get("name") or "—"
            if status == "ACCEPTED":
                lines.append(
                    f"  ACCEPTED epic={alt} ({name}) @ {body['level']} "
                    f"ccy={alt_ccy} dealId={did or '—'}"
                )
                if not accepted_deal_id and did:
                    accepted_deal_id = did
                elif did and did != accepted_deal_id:
                    try:
                        ig.cancel_working_order(did)
                    except Exception:
                        pass
            else:
                lines.append(
                    f"  REJECTED epic={alt} ({name}) reason={reason or '—'} "
                    f"status={snap.get('marketStatus') or '—'} ccy={alt_ccy}"
                )
            ig.config.epic = prev
        except Exception as exc:
            try:
                ig.config.epic = epic
            except Exception:
                pass
            lines.append(f"  ERROR epic={alt}: {type(exc).__name__}: {exc}")

    if accepted_deal_id:
        lines.append("")
        lines.append(
            f"DIAG FOUND working payload/epic — first ACCEPTED dealId={accepted_deal_id}. "
            "Save that epic / payload shape on the connector."
        )
    else:
        lines.append("")
        if attached_on_positions:
            lines.append(
                "DIAG: nothing ACCEPTED, and open position(s) carry attached "
                "stop/limit orders — most likely cause. Close those positions "
                "(or strip their stops/limits) and re-run this test."
            )
        else:
            lines.append(
                "DIAG: nothing ACCEPTED with a flat book. Likely account "
                "entitlement / DEMO restriction on France 40 working orders — "
                "try a fresh IG demo account or contact IG support."
            )
    return {"accepted_deal_id": accepted_deal_id}


def _mid_from_ig_market(ig: Any, epic: str | None = None) -> float:
    """Live mid from GET /markets snapshot (no /prices allowance burn)."""
    market = ig.get_market(epic) if epic else ig.get_market()
    snap = (market.get("snapshot") or {}) if isinstance(market, dict) else {}
    try:
        bid = float(snap.get("bid") or 0)
        offer = float(snap.get("offer") or snap.get("ask") or 0)
    except (TypeError, ValueError):
        bid, offer = 0.0, 0.0
    if bid > 0 and offer > 0:
        mid = (bid + offer) / 2.0
    elif bid > 0:
        mid = bid
    elif offer > 0:
        mid = offer
    else:
        mid = 0.0
    if mid > 0 and hasattr(ig, "ledger"):
        try:
            ig.ledger.last_price = mid
        except Exception:
            pass
    return mid


def _run_ig_market_open_close_probe(ig: Any, *, size: float, currency: str, lines: list[str]) -> bool:
    """Tiny DEMO market open + immediate close. Returns True on success."""
    from chatbot.trader.ig_connector import IgApiError
    from chatbot.trader.models import LegRole, Side

    lines.append("")
    lines.append(
        "Market open/close probe (allow_market_orders=true) — DEMO fees/spread may apply."
    )
    try:
        leg_id = ig.open_market_position(
            Side.BUY, size, role=LegRole.HEDGE, currency=currency
        )
        lines.append(f"MARKET OPEN BUY size={size} · leg={leg_id}")
        ig.market_close(leg_id)
        lines.append(f"MARKET CLOSE leg={leg_id}")
        return True
    except (IgApiError, Exception) as exc:
        lines.append(f"MARKET probe FAILED: {exc}")
        return False


def run_ig_working_order_test(
    config: dict,
    *,
    hold_seconds: float = 5.0,
    allow_market_orders: bool = False,
) -> ConnectorTestResult:
    """
    DEMO-only smoke test: place far working orders, wait, cancel them.

    By default no market open/close. Pass ``allow_market_orders=True`` to also
    open and immediately close a tiny market position after the WO path.
    LIMIT/STOP entries may include an attached ``limitLevel`` (take-profit).
    """
    import httpx
    import time

    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgAuthError, IgConnector, _IG_HOSTS, format_ig_http_error
    from chatbot.trader.models import OrderPurpose, OrderType, Side, WorkingOrder

    api_key = str(config.get("api_key", "")).strip()
    username = str(config.get("username", "")).strip()
    password = str(config.get("password", "")).strip()
    if not api_key or not username or not password:
        return ConnectorTestResult(
            ok=False,
            message="IG API key, username, and password are required (save first, or fill them).",
            error="missing_credentials",
        )
    acc_type = str(config.get("acc_type", "DEMO") or "DEMO").strip().upper()
    if acc_type != "DEMO":
        return ConnectorTestResult(
            ok=False,
            message=(
                "Working-order test is DEMO-only for safety. "
                "Switch Environment to DEMO (and use a Demo API key) before running this."
            ),
            error="live_blocked",
        )
    epic = str(config.get("epic", "") or "IX.D.CAC.BMU.IP").strip()
    account_id = str(config.get("account_id", "")).strip()
    size = float(config.get("order_size") or 1.0)
    if size <= 0:
        size = 1.0
    context = (
        f"env={acc_type}\n"
        f"host={_IG_HOSTS.get(acc_type)}\n"
        f"epic={epic}\n"
        f"account_id={account_id or '(none)'}\n"
        f"hold_seconds={hold_seconds}"
    )
    cfg = TraderConfig(
        ig_api_key=api_key,
        ig_username=username,
        ig_password=password,
        ig_account_id=account_id,
        ig_acc_type=acc_type,
        epic=epic,
        order_size=size,
    )
    ig = IgConnector(cfg, dry_run=False)
    placed: list[WorkingOrder] = []
    lines: list[str] = []
    try:
        ig.login()
        if not ig._cst:
            return ConnectorTestResult(
                ok=False, message="IG login failed (no session tokens).", error="no_session"
            )
        account = ig.get_active_account()
        account_type = str(account.get("accountType") or "").strip().upper() or "—"
        account_ccy = str(
            account.get("currency") or account.get("preferredCurrency") or ""
        ).strip().upper() or "—"
        configured_epic = epic
        epic_hint = ig.epic_product_hint(epic)
        if not ig.epic_compatible_with_account(epic=epic, account_type=account_type):
            lines.append(
                f"Epic/account mismatch: account={account_type} ({account_ccy}) "
                f"but epic {epic} looks like {epic_hint} "
                f"(DAILY/GBP-only = spread bet; CFS/IFS/EUR = CFD)."
            )
            alt, seen = ig.find_compatible_epic(account_type=account_type)
            if alt and alt != epic:
                lines.append(f"Switching test epic {epic} → {alt}")
                epic = alt
                ig.config.epic = alt
                if hasattr(ig, "_market_cache"):
                    ig._market_cache.clear()
            else:
                lines.append(
                    "Could not auto-find a compatible France 40 epic. "
                    "In IG, open France 40 on this CFD account, then copy its epic "
                    "(often IX.D.CAC.BMU.IP / IFS / CFS — not …DAILY.IP)."
                )
                if seen:
                    lines.append(f"Search saw: {', '.join(seen[:12])}")
                lines.append("")
                lines.append(
                    f"env={acc_type}\naccount_type={account_type}\n"
                    f"configured_epic={configured_epic}\nepic_hint={epic_hint}"
                )
                return ConnectorTestResult(
                    ok=False,
                    message="\n".join(lines),
                    error="epic_account_mismatch",
                )

        mid = ig.sync_price()
        if mid <= 0:
            return ConnectorTestResult(
                ok=False, message="Could not read a mid price for the epic.", error="no_price"
            )
        currency = ig.resolve_order_currency()
        min_size = ig.resolve_min_deal_size()
        expiry = ig.resolve_order_expiry()
        allowed_ccy = ig.market_currency_codes()
        market = ig.get_market()
        snapshot = (market.get("snapshot") or {}) if isinstance(market, dict) else {}
        market_status = str(snapshot.get("marketStatus") or "—")
        size = max(size, min_size)
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
        min_dist = ig.resolve_min_stop_or_limit_distance()
        max_dist = ig.resolve_max_stop_or_limit_distance()
        # Prefer a mid-band offset inside IG min/max (price units).
        # Indices: tens of points. FX (mid < 50): small price deltas — never
        # floor at 20 index-points (that snaps EURUSD to nonsense levels).
        fx = mid < 50
        if fx:
            floor = max(float(min_dist or 0) * 2.0, mid * 0.002)
            cap = (max_dist * 0.4) if max_dist > 0 else mid * 0.05
            offset = float(ig.snap_level(min(cap, max(floor, mid * 0.005))))
            if offset <= 0:
                offset = float(ig.snap_level(max(floor, mid * 0.005)))
            tp_offset = float(ig.snap_level(max(offset * 0.5, float(min_dist or 0) * 2.0, mid * 0.002)))
            probe_extras = (
                offset,
                max(offset * 0.5, mid * 0.003),
                max(offset * 1.5, mid * 0.01),
                max(offset * 2.0, mid * 0.02),
                min(max_dist * 0.3, mid * 0.05) if max_dist > 0 else mid * 0.03,
            )
        elif max_dist > 0 and min_dist > 0:
            offset = float(ig.snap_level(min(max_dist * 0.5, max(min_dist * 2.0, 20.0))))
            tp_offset = max(12.0, float(ig.snap_level(offset * 0.5)))
            probe_extras = (15.0, 25.0, 40.0, 60.0, 80.0)
        elif max_dist > 0:
            offset = float(ig.snap_level(min(max_dist * 0.5, 40.0)))
            tp_offset = max(12.0, float(ig.snap_level(offset * 0.5)))
            probe_extras = (15.0, 25.0, 40.0, 60.0, 80.0)
        else:
            offset = max(20.0, float(ig.snap_level(mid * 0.005)))
            tp_offset = max(12.0, float(ig.snap_level(offset * 0.5)))
            probe_extras = (15.0, 25.0, 40.0, 60.0, 80.0)
        buy_entry = ig.snap_level(mid - offset)
        sell_entry = ig.snap_level(mid + offset)
        buy_stop = ig.snap_level(mid + offset)
        sell_stop = ig.snap_level(mid - offset)
        buy_tp = ig.snap_level(max(buy_entry + tp_offset, mid + tp_offset))
        sell_tp = ig.snap_level(min(sell_entry - tp_offset, mid - tp_offset))
        buy_stop_tp = ig.snap_level(max(buy_stop + tp_offset, mid + tp_offset))
        sell_stop_tp = ig.snap_level(min(sell_stop - tp_offset, mid - tp_offset))
        specs: list[tuple[OrderType, Side, float, str, float | None]] = [
            (OrderType.LIMIT, Side.BUY, buy_entry, "BUY LIMIT below (no TP)", None),
            (OrderType.LIMIT, Side.BUY, buy_entry, "BUY LIMIT below + TP", buy_tp),
            (OrderType.STOP, Side.BUY, buy_stop, "BUY STOP above (no TP)", None),
            (OrderType.STOP, Side.BUY, buy_stop, "BUY STOP above + TP", buy_stop_tp),
            (OrderType.LIMIT, Side.SELL, sell_entry, "SELL LIMIT above + TP", sell_tp),
            (OrderType.STOP, Side.SELL, sell_stop, "SELL STOP below (no TP)", None),
            (OrderType.STOP, Side.SELL, sell_stop, "SELL STOP below + TP", sell_stop_tp),
        ]
        # Only probe currencies the market actually lists (plus resolved pick).
        currency_candidates: list[str] = []
        for c in [currency, *allowed_ccy]:
            c = str(c or "").strip().upper()
            if c and c not in currency_candidates:
                currency_candidates.append(c)
        if not currency_candidates:
            currency_candidates = [currency or "EUR"]

        context = (
            f"env={acc_type}\n"
            f"host={_IG_HOSTS.get(acc_type)}\n"
            f"account_type={account_type}\n"
            f"account_currency={account_ccy}\n"
            f"configured_epic={configured_epic}\n"
            f"epic={epic}\n"
            f"epic_hint={ig.epic_product_hint(epic)}\n"
            f"account_id={account_id or '(none)'}\n"
            f"market_status={market_status}\n"
            f"expiry={expiry}\n"
            f"allowed_currencies={','.join(allowed_ccy) or '—'}\n"
            f"currency_tried={','.join(currency_candidates)}\n"
            f"size={size} (minDealSize={min_size})\n"
            f"minNormalStopOrLimitDistance={min_dist}\n"
            f"maxStopOrLimitDistance={max_dist}\n"
            f"dealingRules={rules}\n"
            f"instrument_name={instrument.get('name') or '—'}\n"
            f"stopsLimitsAllowed={instrument.get('stopsLimitsAllowed')}\n"
            f"forceOpenAllowed={instrument.get('forceOpenAllowed')}\n"
            f"controlledRiskAllowed={instrument.get('controlledRiskAllowed')}\n"
            f"hold_seconds={hold_seconds}"
        )
        lines.append(f"IG DEMO working-order test · {epic}")
        if epic != configured_epic:
            lines.append(f"(configured epic was {configured_epic})")
        lines.append(
            f"Account={account_type} {account_ccy} · Mid≈{mid:.5f} · offset={offset:.5f} · "
            f"size={size} · expiry={expiry} · market={market_status}"
        )
        lines.append(
            f"dealingRules minDist={min_dist} maxDist={max_dist} · "
            f"stopsLimitsAllowed={instrument.get('stopsLimitsAllowed')} · "
            f"forceOpenAllowed={instrument.get('forceOpenAllowed')}"
        )
        lines.append(f"Currencies allowed by market: {', '.join(allowed_ccy) or '—'}")
        lines.append("")

        # If the mid-band probe fails, walk several offsets (too far / too close).
        working_currency = currency_candidates[0]
        probe_ok = False
        probe_offsets = [offset]
        for extra in probe_extras:
            try:
                extra_f = float(ig.snap_level(float(extra)))
            except Exception:
                extra_f = float(extra)
            if extra_f <= 0:
                continue
            if all(abs(extra_f - o) > (mid * 1e-4 if fx else 0.5) for o in probe_offsets):
                probe_offsets.append(extra_f)
        last_exc: Exception | None = None
        for off in probe_offsets:
            level = float(ig.snap_level(mid - off))
            if level <= 0:
                lines.append(
                    f"SKIP BUY LIMIT below offset≈{off} → level={level} (<=0; FX scaling?)"
                )
                continue
            for ccy in currency_candidates:
                try:
                    order = ig.place_order(
                        WorkingOrder(
                            id="",
                            type=OrderType.LIMIT,
                            side=Side.BUY,
                            level=level,
                            size=size,
                            purpose=OrderPurpose.ENTRY,
                        ),
                        currency=ccy,
                        limit_level=None,
                    )
                    placed.append(order)
                    working_currency = ccy
                    probe_ok = True
                    lines.append(
                        f"ACCEPTED BUY LIMIT below (no TP) @ {order.level:.5f} "
                        f"(offset≈{off:.5f}) · currency={ccy} "
                        f"· dealId={order.deal_id or '—'} · ref={order.client_ref or '—'}"
                    )
                    # Rebuild remaining specs around the offset that worked.
                    offset = off
                    buy_entry = ig.snap_level(mid - offset)
                    sell_entry = ig.snap_level(mid + offset)
                    buy_stop = ig.snap_level(mid + offset)
                    sell_stop = ig.snap_level(mid - offset)
                    buy_tp = ig.snap_level(max(buy_entry + tp_offset, mid + tp_offset))
                    sell_tp = ig.snap_level(min(sell_entry - tp_offset, mid - tp_offset))
                    buy_stop_tp = ig.snap_level(max(buy_stop + tp_offset, mid + tp_offset))
                    sell_stop_tp = ig.snap_level(min(sell_stop - tp_offset, mid - tp_offset))
                    specs = [
                        (OrderType.LIMIT, Side.BUY, buy_entry, "BUY LIMIT below + TP", buy_tp),
                        (OrderType.STOP, Side.BUY, buy_stop, "BUY STOP above (no TP)", None),
                        (OrderType.STOP, Side.BUY, buy_stop, "BUY STOP above + TP", buy_stop_tp),
                        (OrderType.LIMIT, Side.SELL, sell_entry, "SELL LIMIT above + TP", sell_tp),
                        (OrderType.STOP, Side.SELL, sell_stop, "SELL STOP below (no TP)", None),
                        (OrderType.STOP, Side.SELL, sell_stop, "SELL STOP below + TP", sell_stop_tp),
                    ]
                    break
                except IgAuthError as exc:
                    last_exc = exc
                    lines.append(
                        f"REJECTED BUY LIMIT below (no TP) offset≈{off:.5f} "
                        f"level={level} currency={ccy}: {exc}"
                    )
            if probe_ok:
                break
        if not probe_ok:
            lines.append("")
            lines.append(
                "Could not place a bare BUY LIMIT at any tried offset. "
                "Running payload / shape / epic diagnostics…"
            )
            if last_exc is not None:
                lines.append(f"Last error: {last_exc}")
            diag = _diagnose_ig_working_order_rejects(
                ig,
                mid=mid,
                size=size,
                currency=working_currency,
                expiry=expiry,
                epic=epic,
                min_dist=float(min_dist or 12.0),
                lines=lines,
            )
            if diag.get("accepted_deal_id"):
                placed_deal = str(diag["accepted_deal_id"])
                try:
                    ig.cancel_working_order(placed_deal)
                    lines.append(f"Cancelled diagnostic dealId={placed_deal}")
                except Exception as exc:
                    lines.append(f"Cancel FAILED diagnostic dealId={placed_deal}: {exc}")
            lines.append("")
            lines.append(context)
            return ConnectorTestResult(ok=False, message="\n".join(lines), error=context)

        accepted = 1
        rejected = 0
        attached_tp_expected = 0
        for otype, side, level, label, tp_level in specs:
            try:
                order = ig.place_order(
                    WorkingOrder(
                        id="",
                        type=otype,
                        side=side,
                        level=float(level),
                        size=size,
                        purpose=OrderPurpose.ENTRY,
                    ),
                    currency=working_currency,
                    limit_level=tp_level,
                )
                placed.append(order)
                accepted += 1
                if tp_level is not None:
                    attached_tp_expected += 1
                tp_bit = f" · TP@{tp_level:.2f}" if tp_level is not None else ""
                lines.append(
                    f"ACCEPTED {label} @ {order.level:.2f}{tp_bit} · currency={working_currency} "
                    f"· dealId={order.deal_id or '—'} · ref={order.client_ref or '—'}"
                )
            except IgAuthError as exc:
                rejected += 1
                lines.append(f"REJECTED {label} @ {level:.2f}: {exc}")
        lines.append("")
        lines.append(f"Using currency={working_currency} for this test.")
        lines.append(
            "LIMIT orders use IG attached limitLevel (take-profit arms when the entry fills)."
        )
        lines.append("")
        open_orders = ig.list_working_orders()
        lines.append(f"Open working orders on account now: {len(open_orders)}")
        attached_tp_seen = 0
        for row in open_orders:
            lim = row.get("limitLevel")
            stp = row.get("stopLevel")
            if lim is not None and str(lim).strip() not in ("", "None", "0", "0.0"):
                attached_tp_seen += 1
            lim_s = f" limitLevel={lim}" if lim is not None else ""
            stp_s = f" stopLevel={stp}" if stp is not None else ""
            lines.append(
                f"  · dealId={row.get('dealId') or '—'} "
                f"{row.get('direction') or ''} {row.get('orderType') or row.get('type') or ''} "
                f"@ {row.get('orderLevel') or row.get('level') or '—'} "
                f"epic={row.get('epic') or '—'}{lim_s}{stp_s}"
            )
        if not open_orders:
            lines.append("  (none)")
        lines.append(
            f"Attached TP on open WOs: seen={attached_tp_seen} "
            f"(expected≥{attached_tp_expected} for the LIMIT entries)"
        )
        lines.append("")
        lines.append(f"Holding {hold_seconds:.0f}s before cancel…")
        time.sleep(max(0.5, float(hold_seconds)))
        lines.append("")
        open_after = ig.list_working_orders()
        cancel_ids = [
            str(row.get("dealId") or "").strip()
            for row in open_after
            if str(row.get("dealId") or "").strip()
        ]
        if not cancel_ids:
            cancel_ids = [o.deal_id for o in placed if o.deal_id]
        cancel_ok = 0
        cancel_fail = 0
        if not cancel_ids:
            lines.append("Nothing to cancel (no open working orders / no dealIds).")
        for deal_id in cancel_ids:
            try:
                ig.cancel_working_order(deal_id)
                cancel_ok += 1
                lines.append(f"Cancelled dealId={deal_id}")
            except Exception as exc:
                cancel_fail += 1
                lines.append(f"Cancel FAILED dealId={deal_id}: {exc}")
        remaining = ig.list_working_orders()
        lines.append("")
        lines.append(
            f"Done · accepted={accepted} · rejected={rejected} · "
            f"cancelled={cancel_ok} · cancel_failed={cancel_fail} · "
            f"still_open={len(remaining)} · attached_tp_seen={attached_tp_seen}"
        )
        market_ok = True
        if allow_market_orders:
            market_ok = _run_ig_market_open_close_probe(
                ig, size=size, currency=working_currency, lines=lines
            )
        else:
            lines.append("Market open/close skipped (allow_market_orders=false).")
        lines.append("")
        lines.append(context)
        # Soft-fail if IG accepted LIMIT+TP but list omitted limitLevel (some envs
        # hide it until fill). Hard-fail only when place/cancel themselves fail.
        ok = (
            accepted > 0
            and rejected == 0
            and cancel_fail == 0
            and len(remaining) == 0
            and market_ok
        )
        if ok and attached_tp_expected > 0 and attached_tp_seen < 1:
            lines.append("")
            lines.append(
                "Note: LIMIT+TP places were ACCEPTED but limitLevel was not visible "
                "on GET /workingorders — check the IG Working Orders ticket for "
                "attached profit. Not treated as a hard failure."
            )
        return ConnectorTestResult(ok=ok, message="\n".join(lines), error=None if ok else context)
    except IgAuthError as exc:
        return ConnectorTestResult(ok=False, message=str(exc), error=context)
    except httpx.HTTPStatusError as exc:
        detail = format_ig_http_error(exc.response, action="request", url=str(exc.request.url))
        return ConnectorTestResult(ok=False, message=detail, error=context)
    except httpx.RequestError as exc:
        return ConnectorTestResult(ok=False, message=f"IG network error: {exc}", error=context)
    except Exception as exc:
        try:
            for row in ig.list_working_orders():
                did = str(row.get("dealId") or "").strip()
                if did:
                    try:
                        ig.cancel_working_order(did)
                    except Exception:
                        pass
        except Exception:
            pass
        for order in placed:
            if order.deal_id:
                try:
                    ig.cancel_working_order(order.deal_id)
                except Exception:
                    pass
        return ConnectorTestResult(ok=False, message=f"Working-order test failed: {exc}", error=context)
    finally:
        ig.close()


def run_ig_cac40_working_order_matrix(
    config: dict,
    *,
    allow_market_orders: bool = False,
    use_stream_confirms: bool = True,
    stream_timeout: float = 20.0,
) -> ConnectorTestResult:
    """France CAC DEMO matrix: LIMIT±TP and STOP±TP; cancel each; market only if flagged.

    When ``use_stream_confirms`` is True, also connects Lightstreamer TRADE and
    records whether a WOU/CONFIRMS update arrived (REST confirm remains authoritative).
    """
    import json
    import time

    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgApiError, IgAuthError, IgConnector
    from chatbot.trader.models import OrderPurpose, OrderType, Side, WorkingOrder

    api_key = str(config.get("api_key", "")).strip()
    username = str(config.get("username", "")).strip()
    password = str(config.get("password", "")).strip()
    if not api_key or not username or not password:
        return ConnectorTestResult(
            ok=False,
            message="IG API key, username, and password are required.",
            error="missing_credentials",
        )
    acc_type = str(config.get("acc_type", "DEMO") or "DEMO").strip().upper()
    if acc_type != "DEMO":
        return ConnectorTestResult(
            ok=False,
            message="France CAC working-order matrix is DEMO-only.",
            error="live_blocked",
        )
    epic = str(config.get("epic", "") or "IX.D.CAC.BMU.IP").strip() or "IX.D.CAC.BMU.IP"
    account_id = str(config.get("account_id", "")).strip()
    size = float(config.get("order_size") or 1.0) or 1.0
    cfg = TraderConfig(
        ig_api_key=api_key,
        ig_username=username,
        ig_password=password,
        ig_account_id=account_id,
        ig_acc_type=acc_type,
        epic=epic,
        order_size=size,
    )
    ig = IgConnector(cfg, dry_run=False)
    lines: list[str] = []
    ls_session = None
    case_ok: dict[str, bool] = {}
    try:
        ig.login()
        if not ig._cst:
            return ConnectorTestResult(
                ok=False, message="IG login failed (no session tokens).", error="no_session"
            )
        account = ig.get_active_account()
        account_type = str(account.get("accountType") or "").strip().upper() or "—"
        if not ig.epic_compatible_with_account(epic=epic, account_type=account_type):
            alt, _seen = ig.find_compatible_epic(account_type=account_type)
            if alt and alt != epic:
                lines.append(f"Switching epic {epic} → {alt}")
                epic = alt
                ig.config.epic = alt
                if hasattr(ig, "_market_cache"):
                    ig._market_cache.clear()
            else:
                return ConnectorTestResult(
                    ok=False,
                    message="No compatible France 40 epic for this account.",
                    error="epic_account_mismatch",
                )
        # Prefer market snapshot — avoids burning weekly /prices allowance.
        mid = _mid_from_ig_market(ig)
        if mid <= 0:
            return ConnectorTestResult(
                ok=False, message="Could not read mid price from market snapshot.", error="no_price"
            )
        currency = ig.resolve_order_currency()
        size = max(size, ig.resolve_min_deal_size())
        min_dist = float(ig.resolve_min_stop_or_limit_distance() or 12.0)
        max_dist = float(ig.resolve_max_stop_or_limit_distance() or 0.0)
        if max_dist > 0 and min_dist > 0:
            offset = float(ig.snap_level(min(max_dist * 0.5, max(min_dist * 2.0, 20.0))))
        else:
            offset = max(20.0, float(ig.snap_level(mid * 0.005)))
        tp_offset = max(12.0, float(ig.snap_level(offset * 0.5)))
        buy_limit = float(ig.snap_level(mid - offset))
        buy_stop = float(ig.snap_level(mid + offset))
        buy_limit_tp = float(ig.snap_level(max(buy_limit + tp_offset, mid + tp_offset)))
        buy_stop_tp = float(ig.snap_level(max(buy_stop + tp_offset, mid + tp_offset)))
        matrix: list[tuple[str, OrderType, float, float | None]] = [
            ("LIMIT_no_TP", OrderType.LIMIT, buy_limit, None),
            ("LIMIT_with_TP", OrderType.LIMIT, buy_limit, buy_limit_tp),
            ("STOP_no_TP", OrderType.STOP, buy_stop, None),
            ("STOP_with_TP", OrderType.STOP, buy_stop, buy_stop_tp),
        ]
        if use_stream_confirms and ig.lightstreamer_endpoint and (
            ig.current_account_id or account_id
        ):
            try:
                from chatbot.trader.ig_stream_probe import IgLightstreamerSession

                ls_session = IgLightstreamerSession(
                    endpoint=ig.lightstreamer_endpoint,
                    account_id=(ig.current_account_id or account_id).strip(),
                    cst=ig._cst or "",
                    xst=ig._security or "",
                )
                if ls_session.connect(timeout=15.0):
                    ls_session.subscribe_trade()
                    lines.append(f"Lightstreamer TRADE subscribed · status={ls_session.status}")
                else:
                    lines.append("Lightstreamer connect failed — REST confirms only.")
                    ls_session.disconnect()
                    ls_session = None
            except Exception as exc:
                lines.append(f"Lightstreamer unavailable ({exc}) — REST confirms only.")
                ls_session = None

        lines.append(f"France CAC working-order matrix · {epic}")
        lines.append(
            f"Account={account_type} · Mid≈{mid:.2f} · offset={offset:.2f} · "
            f"size={size} · currency={currency}"
        )
        lines.append("")

        for case_id, otype, level, tp in matrix:
            stream_hit = "n/a"
            try:
                before_trade = ls_session.trade.count if ls_session else 0
                order = ig.place_order(
                    WorkingOrder(
                        id="",
                        type=otype,
                        side=Side.BUY,
                        level=level,
                        size=size,
                        purpose=OrderPurpose.ENTRY,
                    ),
                    currency=currency,
                    limit_level=tp,
                )
                deal_id = order.deal_id or ""
                if ls_session and deal_id:
                    hit = ls_session.trade.wait_for(
                        lambda row, did=deal_id: did in json.dumps(row, default=str),
                        timeout=stream_timeout,
                    )
                    stream_hit = "yes" if hit or ls_session.trade.count > before_trade else "no"
                tp_bit = f" TP@{tp:.2f}" if tp is not None else ""
                lines.append(
                    f"PASS {case_id} BUY {otype.value} @ {order.level:.2f}{tp_bit} "
                    f"dealId={deal_id or '—'} stream={stream_hit}"
                )
                if deal_id:
                    ig.cancel_working_order(deal_id)
                    lines.append(f"  cancelled {deal_id}")
                case_ok[case_id] = True
            except (IgAuthError, IgApiError, Exception) as exc:
                case_ok[case_id] = False
                lines.append(f"FAIL {case_id}: {exc}")
            time.sleep(0.2)

        market_ok = True
        if allow_market_orders:
            market_ok = _run_ig_market_open_close_probe(
                ig, size=size, currency=currency, lines=lines
            )
        else:
            lines.append("Market open/close skipped (allow_market_orders=false).")

        all_cases = all(case_ok.get(c[0], False) for c in matrix)
        ok = all_cases and market_ok
        lines.append("")
        lines.append(
            f"Summary · cases={case_ok} · market_ok={market_ok} · "
            f"allow_market_orders={allow_market_orders}"
        )
        return ConnectorTestResult(
            ok=ok,
            message="\n".join(lines),
            error=None if ok else "matrix_failed",
        )
    except Exception as exc:
        return ConnectorTestResult(
            ok=False, message=f"CAC matrix failed: {exc}", error="exception"
        )
    finally:
        if ls_session is not None:
            try:
                ls_session.disconnect()
            except Exception:
                pass
        ig.close()


def run_ig_stream_order_probe(
    config: dict,
    *,
    allow_market_orders: bool = False,
) -> ConnectorTestResult:
    """Orchestrate CAC matrix (+ optional EURUSD smoke) over Lightstreamer TRADE."""
    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgConnector
    from chatbot.trader.models import OrderPurpose, OrderType, Side, WorkingOrder

    cac = run_ig_cac40_working_order_matrix(
        config,
        allow_market_orders=allow_market_orders,
        use_stream_confirms=True,
    )
    lines = [cac.message, "", "--- EURUSD Mini smoke (LIMIT/STOP bare) ---"]
    # Still probe FX when CAC is rejected by IG DEMO (common ATTACHED_ORDER_LEVEL_ERROR).

    api_key = str(config.get("api_key", "")).strip()
    username = str(config.get("username", "")).strip()
    password = str(config.get("password", "")).strip()
    account_id = str(config.get("account_id", "")).strip()
    fx_epic = "CS.D.EURUSD.MINI.IP"
    cfg = TraderConfig(
        ig_api_key=api_key,
        ig_username=username,
        ig_password=password,
        ig_account_id=account_id,
        ig_acc_type="DEMO",
        epic=fx_epic,
        order_size=float(config.get("order_size") or 1.0) or 1.0,
    )
    ig = IgConnector(cfg, dry_run=False)
    fx_ok = True
    try:
        ig.login()
        mid = _mid_from_ig_market(ig)
        if mid <= 0:
            lines.append("SKIP EURUSD — no mid from market snapshot")
            fx_ok = False
        else:
            currency = ig.resolve_order_currency()
            size = max(float(cfg.order_size), ig.resolve_min_deal_size())
            min_d = float(ig.resolve_min_stop_or_limit_distance() or 0) or mid * 0.001
            offset = max(min_d * 2.0, mid * 0.005)
            for label, otype, level in (
                ("FX LIMIT bare", OrderType.LIMIT, mid - offset),
                ("FX STOP bare", OrderType.STOP, mid + offset),
            ):
                try:
                    order = ig.place_order(
                        WorkingOrder(
                            id="",
                            type=otype,
                            side=Side.BUY,
                            level=float(ig.snap_level(level)),
                            size=size,
                            purpose=OrderPurpose.ENTRY,
                        ),
                        currency=currency,
                        limit_level=None,
                    )
                    lines.append(
                        f"PASS {label} @ {order.level:.5f} dealId={order.deal_id or '—'}"
                    )
                    if order.deal_id:
                        ig.cancel_working_order(order.deal_id)
                except Exception as exc:
                    fx_ok = False
                    lines.append(f"FAIL {label}: {exc}")
    except Exception as exc:
        fx_ok = False
        lines.append(f"FAIL EURUSD smoke: {exc}")
    finally:
        ig.close()

    ok = cac.ok and fx_ok
    return ConnectorTestResult(
        ok=ok, message="\n".join(lines), error=None if ok else "stream_order_probe_failed"
    )


def run_ig_cross_market_working_order_probe(
    config: dict,
    *,
    allow_market_orders: bool = False,
) -> ConnectorTestResult:
    """DEMO-only: bare LIMIT on configured epic + alternate TRADEABLE markets.

    Separate from ``run_ig_working_order_test`` (no attached TP, no France-40-only
    diagnostics). Used to check whether ``ATTACHED_ORDER_LEVEL_ERROR`` is
    instrument-specific or account/API-wide.
    """
    import httpx

    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgApiError, IgAuthError, IgConnector, _IG_HOSTS, format_ig_http_error

    api_key = str(config.get("api_key", "")).strip()
    username = str(config.get("username", "")).strip()
    password = str(config.get("password", "")).strip()
    if not api_key or not username or not password:
        return ConnectorTestResult(
            ok=False,
            message="IG API key, username, and password are required.",
            error="missing_credentials",
        )
    acc_type = str(config.get("acc_type", "DEMO") or "DEMO").strip().upper()
    if acc_type != "DEMO":
        return ConnectorTestResult(
            ok=False,
            message="Cross-market working-order probe is DEMO-only.",
            error="live_blocked",
        )
    configured_epic = str(config.get("epic", "") or "IX.D.CAC.BMU.IP").strip()
    account_id = str(config.get("account_id", "")).strip()
    size_override = float(config.get("order_size") or 0) or None
    cfg = TraderConfig(
        ig_api_key=api_key,
        ig_username=username,
        ig_password=password,
        ig_account_id=account_id,
        ig_acc_type=acc_type,
        epic=configured_epic,
        order_size=float(size_override or 1.0),
    )
    ig = IgConnector(cfg, dry_run=False)
    lines: list[str] = []
    accepted_any = False
    context = (
        f"env={acc_type}\n"
        f"host={_IG_HOSTS.get(acc_type)}\n"
        f"account_id={account_id or '(none)'}\n"
        f"configured_epic={configured_epic}"
    )

    def _mid_from_market(market: dict[str, Any]) -> tuple[float, float, float, str]:
        snap = (market.get("snapshot") or {}) if isinstance(market, dict) else {}
        bid = float(snap.get("bid") or 0)
        offer = float(snap.get("offer") or snap.get("ask") or 0)
        mid = (bid + offer) / 2.0 if bid > 0 and offer > 0 else float(snap.get("netChange") or 0)
        if mid <= 0 and bid > 0:
            mid = bid
        status = str(snap.get("marketStatus") or "—")
        return bid, offer, mid, status

    def _min_dist(market: dict[str, Any], mid: float) -> float:
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
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
        rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
        row = rules.get("minDealSize") or {}
        try:
            value = float(row.get("value") or 0)
        except (TypeError, ValueError):
            value = 0.0
        if size_override and size_override > 0:
            return max(size_override, value or size_override)
        return value if value > 0 else 1.0

    def _pick_currency(market: dict[str, Any], account_ccy: str) -> str:
        instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
        codes: list[str] = []
        for row in instrument.get("currencies") or []:
            if isinstance(row, dict) and row.get("code"):
                code = str(row["code"]).strip().upper()
                if code and code not in codes:
                    codes.append(code)
        if account_ccy and account_ccy in codes:
            return account_ccy
        for prefer in ("EUR", "USD", "GBP"):
            if prefer in codes:
                return prefer
        return codes[0] if codes else (account_ccy or "EUR")

    def _expiry(market: dict[str, Any], epic: str) -> str:
        instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
        expiry = str(instrument.get("expiry") or "").strip()
        if expiry and expiry not in ("null", "None"):
            return expiry
        return "DFB" if ig.epic_product_hint(epic, market=market) == "SPREADBET" else "-"

    def _price_decimals(mid: float) -> int:
        if mid < 10:
            return 5
        if mid < 500:
            return 2
        return 1

    def _entry_offset(mid: float, min_d: float) -> float:
        # Keep level on the correct side of market and above zero for FX.
        if mid < 50:
            return max(min(min_d, mid * 0.05) * 2.0, mid * 0.005)
        return max(min_d * 2.0, 25.0)

    def _try_wo(
        *,
        epic: str,
        label: str,
        order_type: str,
        with_tp: bool,
    ) -> None:
        """Place one BUY working order (LIMIT below / STOP above), optional attached TP."""
        nonlocal accepted_any
        order_type = order_type.upper()
        shape = f"BUY {order_type}{' +TP' if with_tp else ''}"
        try:
            if hasattr(ig, "_market_cache"):
                ig._market_cache.pop(epic, None)
            prev = ig.config.epic
            ig.config.epic = epic
            market = ig.get_market(epic)
            bid, offer, mid, status = _mid_from_market(market)
            if mid <= 0 or status != "TRADEABLE":
                lines.append(
                    f"  SKIP {label} {shape} epic={epic} status={status} mid={mid:.4f}"
                )
                ig.config.epic = prev
                return
            min_d = _min_dist(market, mid)
            size = _min_size(market)
            ccy = _pick_currency(market, account_ccy)
            expiry = _expiry(market, epic)
            offset = _entry_offset(mid, min_d)
            decimals = _price_decimals(mid)
            if order_type == "LIMIT":
                level = round(mid - offset, decimals)
            else:
                level = round(mid + offset, decimals)
            if level <= 0:
                lines.append(
                    f"  SKIP {label} {shape} epic={epic} level={level} <= 0 "
                    f"(mid={mid}, minDist={min_d})"
                )
                ig.config.epic = prev
                return
            # BUY TP: FX CFD with maxStop in POINTS → limitDistance (capped).
            # Otherwise absolute limitLevel cleared above live offer.
            if with_tp:
                rules = (market.get("dealingRules") or {}) if isinstance(market, dict) else {}
                mx = rules.get("maxStopOrLimitDistance") or {}
                mn = rules.get("minNormalStopOrLimitDistance") or {}
                try:
                    max_v = float(mx.get("value") or 0)
                except (TypeError, ValueError):
                    max_v = 0.0
                try:
                    raw_min = float(mn.get("value") or 0)
                except (TypeError, ValueError):
                    raw_min = 0.0
                max_u = str(mx.get("unit") or "POINTS").upper()
                if mid < 50 and max_u == "POINTS" and max_v > 0:
                    lo = max(raw_min * 2.0 if raw_min > 0 else 10.0, 10.0)
                    hi = max_v * 0.8 if max_v > lo else max_v
                    dist = min(max(lo, 20.0), hi) if hi > 0 else 20.0
                    limit_level = None
                    limit_distance: float | None = float(round(dist, 2))
                else:
                    if mid < 50:
                        clear = max(offset, mid * 0.01)
                    elif mid < 1000:
                        clear = max(offset, min_d * 2.0, mid * 0.02)
                    else:
                        clear = max(offset, min_d * 2.0, 25.0)
                    live_ref = offer if offer > 0 else mid
                    limit_level = round(max(level + clear, live_ref + clear), decimals)
                    limit_distance = None
            else:
                limit_level = None
                limit_distance = None
            name = str(
                ((market.get("instrument") or {}) if isinstance(market, dict) else {}).get("name")
                or ""
            )
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
            if limit_level is not None:
                body["limitLevel"] = limit_level
            if limit_distance is not None:
                body["limitDistance"] = limit_distance
            lines.append(
                f"  TRY {label} {shape} {name or epic} mid={mid:.4f} "
                f"level={level} tp={limit_level if limit_level is not None else '—'} "
                f"tpDist={limit_distance if limit_distance is not None else '—'} "
                f"size={size} ccy={ccy} expiry={expiry} minDist≈{min_d:.4f}"
            )
            conf = ig.submit_working_order_raw(body, version="2")
            status_c = str(conf.get("dealStatus") or "").upper()
            reason = str(conf.get("reason") or "")
            did = str(conf.get("dealId") or "")
            ref = str(conf.get("dealReference") or "")
            if status_c == "ACCEPTED":
                accepted_any = True
                lines.append(f"  ACCEPTED {label} {shape} dealId={did} dealReference={ref}")
                if did:
                    try:
                        ig.cancel_working_order(did)
                        lines.append(f"  cancelled {did}")
                    except Exception as cancel_exc:
                        lines.append(f"  cancel FAILED {did}: {cancel_exc}")
            else:
                lines.append(
                    f"  REJECTED {label} {shape} reason={reason or '—'} "
                    f"dealStatus={status_c or '—'} dealId={did or '—'} "
                    f"dealReference={ref or '—'}"
                )
            ig.config.epic = prev
        except IgApiError as exc:
            lines.append(f"  ERROR {label} {shape} epic={epic}: {exc}")
            try:
                ig.config.epic = configured_epic
            except Exception:
                pass
        except Exception as exc:
            lines.append(f"  ERROR {label} {shape} epic={epic}: {type(exc).__name__}: {exc}")
            try:
                ig.config.epic = configured_epic
            except Exception:
                pass

    def _probe_shapes(*, epic: str, label: str) -> None:
        for order_type in ("LIMIT", "STOP"):
            for with_tp in (False, True):
                _try_wo(epic=epic, label=label, order_type=order_type, with_tp=with_tp)

    def _resolve_search_epic(term: str, *, account_type: str) -> tuple[str, str] | None:
        try:
            rows = ig.search_markets(term)
        except Exception as exc:
            lines.append(f"  search '{term}' failed: {exc}")
            return None

        def _rank(row: dict[str, Any]) -> tuple[int, str]:
            epic = str(row.get("epic") or "")
            name = str(row.get("instrumentName") or row.get("instrument") or "").upper()
            # Prefer US cash share CFD over ITA listings / knock-outs.
            score = 50
            if "ITA" in name:
                score += 100
            if "BULL" in epic.upper() or "BEAR" in epic.upper() or "KO" in name:
                score += 80
            if epic.upper().startswith("UD.D.") or epic.upper().startswith("UA.D."):
                score -= 20
            if "CASH" in epic.upper():
                score -= 10
            return (score, epic)

        eligible: list[dict[str, Any]] = []
        for row in rows:
            epic = str(row.get("epic") or "").strip()
            if not epic:
                continue
            status = str(row.get("marketStatus") or "").upper()
            if status and status != "TRADEABLE":
                continue
            hint = ig.epic_product_hint(epic)
            if account_type == "CFD" and hint == "SPREADBET":
                continue
            if account_type == "SPREADBET" and hint == "CFD":
                continue
            eligible.append(row)
        if not eligible:
            for row in rows:
                epic = str(row.get("epic") or "").strip()
                status = str(row.get("marketStatus") or "").upper()
                if epic and (not status or status == "TRADEABLE"):
                    eligible.append(row)
        if not eligible:
            return None
        best = sorted(eligible, key=_rank)[0]
        epic = str(best.get("epic") or "").strip()
        name = str(best.get("instrumentName") or best.get("instrument") or epic)
        return epic, name

    account_ccy = "—"
    try:
        ig.login()
        if not ig._cst:
            return ConnectorTestResult(
                ok=False, message="IG login failed (no session tokens).", error="no_session"
            )
        account = ig.get_active_account()
        account_type = str(account.get("accountType") or "").strip().upper() or "—"
        account_ccy = str(
            account.get("currency") or account.get("preferredCurrency") or ""
        ).strip().upper() or "—"
        lines.append("IG DEMO cross-market working-order probe")
        lines.append(
            f"Account={account_type} {account_ccy} · configured_epic={configured_epic}"
        )
        try:
            n_pos = len(ig.list_open_positions(epic=""))
            n_wo = len(ig.list_working_orders())
            lines.append(f"Book: open_positions={n_pos} open_working_orders={n_wo}")
        except Exception as exc:
            lines.append(f"Book: unavailable ({exc})")

        lines.append("")
        lines.append("1) Configured epic — BUY LIMIT / STOP ± TP:")
        _probe_shapes(epic=configured_epic, label="configured")

        lines.append("")
        lines.append("2) Forex / Tesla / Gold — BUY LIMIT / STOP ± TP:")
        targets: list[tuple[str, str]] = []
        for term, fallbacks in (
            ("EUR/USD", ("CS.D.EURUSD.MINI.IP", "CS.D.EURUSD.CFD.IP")),
            ("Tesla", ("UD.D.TSLA.CASH.IP", "UA.D.TSLA.CASH.IP")),
            ("Gold", ("CS.D.IN_GOLD.MFI.IP", "CS.D.CFDGOLD.BMU.IP")),
        ):
            resolved = _resolve_search_epic(term, account_type=account_type)
            if resolved:
                targets.append((resolved[0], f"'{term}'"))
            else:
                for fb in fallbacks:
                    targets.append((fb, f"{term} (fallback)"))
                    break

        seen_epics = {configured_epic}
        for epic, label in targets:
            if epic in seen_epics:
                continue
            seen_epics.add(epic)
            lines.append(f"--- {label} epic={epic} ---")
            _probe_shapes(epic=epic, label=label)

        lines.append("")
        if accepted_any:
            lines.append(
                "RESULT: at least one working order was ACCEPTED — "
                "compare shapes/markets above (not a total API ban)."
            )
        else:
            lines.append(
                "RESULT: every LIMIT/STOP (±TP) rejected/skipped on probed markets."
            )
        market_ok = True
        if allow_market_orders:
            ig.config.epic = configured_epic
            try:
                size = max(float(size_override or 1.0), ig.resolve_min_deal_size())
                ccy = ig.resolve_order_currency()
            except Exception:
                size = float(size_override or 1.0) or 1.0
                ccy = account_ccy if account_ccy != "—" else "EUR"
            market_ok = _run_ig_market_open_close_probe(
                ig, size=size, currency=ccy, lines=lines
            )
        else:
            lines.append("Market open/close skipped (allow_market_orders=false).")
        lines.append("")
        lines.append(context)
        ok = accepted_any and market_ok
        return ConnectorTestResult(
            ok=ok,
            message="\n".join(lines),
            error=None if ok else context,
        )
    except IgAuthError as exc:
        return ConnectorTestResult(ok=False, message=str(exc), error=context)
    except httpx.HTTPStatusError as exc:
        detail = format_ig_http_error(exc.response, action="request", url=str(exc.request.url))
        return ConnectorTestResult(ok=False, message=detail, error=context)
    except httpx.RequestError as exc:
        return ConnectorTestResult(ok=False, message=f"IG network error: {exc}", error=context)
    except Exception as exc:
        return ConnectorTestResult(
            ok=False, message=f"Cross-market probe failed: {exc}", error=context
        )
    finally:
        ig.close()


def run_mail_connection_test(
    connection: MailConnection,
    *,
    test: str,
    session,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    resolved_settings = settings or get_settings()
    svc = MailConnectionService(session)
    try:
        if test == "imap":
            mail_cfg, _updated = svc.resolve_runtime_config(
                connection, direction="in", settings=resolved_settings
            )
            return _test_imap(mail_cfg)
        if test == "smtp":
            mail_cfg, _updated = svc.resolve_runtime_config(
                connection, direction="out", settings=resolved_settings
            )
            return _test_outbound_smtp(mail_cfg)
        return ConnectorTestResult(ok=False, message="Invalid test type.", error="invalid_test")
    except (ImapError, EmailSendError, MailOAuthError) as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))
    except Exception as exc:
        return ConnectorTestResult(ok=False, message="Connection failed.", error=str(exc))


def _test_imap(
    config: dict,
    *,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    mail_cfg = _mail_config_for_test(
        config, direction="in", session=session, tenant_id=tenant_id, settings=settings
    )
    client = ImapMailClient(mail_cfg)
    try:
        client.connect()
    finally:
        client.close()
    auth_label = "OAuth" if is_oauth_auth_type(resolve_email_auth_type(config)) else "password"
    return ConnectorTestResult(
        ok=True,
        message=f"IMAP connection OK ({auth_label}) — INBOX accessible.",
    )


def _test_outbound(
    config: dict,
    *,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> ConnectorTestResult:
    provider = resolve_email_outbound_provider(config)
    if provider == EmailOutboundProvider.SMTP.value:
        mail_cfg = _mail_config_for_test(
            config, direction="out", session=session, tenant_id=tenant_id, settings=settings
        )
        return _test_outbound_smtp(mail_cfg, source_config=config)
    sender = build_email_sender(config)
    sender.verify_connection()
    return ConnectorTestResult(ok=True, message=f"{provider.capitalize()} connection OK.")


def _smtp_access_token(mail_cfg: dict) -> str | None:
    token = (
        str(mail_cfg.get("_resolved_access_token", "")).strip()
        or str(mail_cfg.get("oauth_access_token", "")).strip()
    )
    return token or None


def _test_outbound_smtp(mail_cfg: dict, *, source_config: dict | None = None) -> ConnectorTestResult:
    host = str(mail_cfg.get("smtp_host", "")).strip()
    if not host:
        raise EmailSendError("Missing smtp_host")
    port_raw = str(mail_cfg.get("smtp_port", "587")).strip() or "587"
    port = int(port_raw)
    cfg = source_config or mail_cfg
    auth_type = resolve_email_auth_type(cfg)
    access_token = _smtp_access_token(mail_cfg)
    if is_oauth_auth_type(auth_type) and not access_token:
        raise EmailSendError("OAuth access token is missing — reconnect the mailbox.")
    SmtpEmailSender(
        host=host,
        port=port,
        username=str(mail_cfg.get("smtp_username", "")).strip(),
        password=str(mail_cfg.get("smtp_password", "")).strip(),
        use_tls=_parse_use_tls(mail_cfg.get("smtp_use_tls"), default=True),
        access_token=access_token,
    ).verify_connection()
    auth_label = "OAuth" if is_oauth_auth_type(auth_type) else "password"
    return ConnectorTestResult(ok=True, message=f"SMTP connection OK ({auth_label}).")


def _mail_config_for_test(
    config: dict,
    *,
    direction: str,
    session=None,
    tenant_id: int | None = None,
    settings: Settings | None = None,
) -> dict:
    resolved_settings = settings or get_settings()
    raw_id = config.get("mail_connection_id")
    if raw_id is not None and str(raw_id).strip() != "" and session is not None and tenant_id is not None:
        connection = MailConnectionService(session).get_for_tenant(int(raw_id), tenant_id)
        if connection is not None:
            mail_cfg, _updated = MailConnectionService(session).resolve_runtime_config(
                connection,
                direction=direction,
                settings=resolved_settings,
            )
            return mail_cfg
    auth_type = resolve_email_auth_type(config)
    if not is_oauth_auth_type(auth_type):
        return config
    mail_cfg, _updated = prepare_oauth_mail_config(
        config,
        direction=direction,
        settings=resolved_settings,
    )
    return mail_cfg
