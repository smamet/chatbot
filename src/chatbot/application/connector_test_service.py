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

    from chatbot.cac40.config import Cac40Config
    from chatbot.cac40.ig_connector import IgAuthError, IgConnector, _IG_HOSTS, format_ig_http_error

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
    cfg = Cac40Config(
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


def run_ig_working_order_test(config: dict, *, hold_seconds: float = 5.0) -> ConnectorTestResult:
    """
    DEMO-only smoke test: place far working orders, wait, cancel them.

    No market open/close — only LIMIT/STOP working orders away from mid.
    """
    import httpx
    import time

    from chatbot.cac40.config import Cac40Config
    from chatbot.cac40.ig_connector import IgAuthError, IgConnector, _IG_HOSTS, format_ig_http_error
    from chatbot.cac40.models import OrderPurpose, OrderType, Side, WorkingOrder

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
    cfg = Cac40Config(
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
        offset = max(80.0, float(ig.snap_level(mid * 0.01)))
        specs = [
            (OrderType.LIMIT, Side.BUY, ig.snap_level(mid - offset), "BUY LIMIT below"),
            (OrderType.LIMIT, Side.SELL, ig.snap_level(mid + offset), "SELL LIMIT above"),
            (OrderType.STOP, Side.BUY, ig.snap_level(mid + offset), "BUY STOP above"),
            (OrderType.STOP, Side.SELL, ig.snap_level(mid - offset), "SELL STOP below"),
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
            f"hold_seconds={hold_seconds}"
        )
        lines.append(f"IG DEMO working-order test · {epic}")
        if epic != configured_epic:
            lines.append(f"(configured epic was {configured_epic})")
        lines.append(
            f"Account={account_type} {account_ccy} · Mid≈{mid:.2f} · offset={offset:.1f} · "
            f"size={size} · expiry={expiry} · market={market_status}"
        )
        lines.append(f"Currencies allowed by market: {', '.join(allowed_ccy) or '—'}")
        lines.append("")

        working_currency = currency_candidates[0]
        probe_type, probe_side, probe_level, probe_label = specs[0]
        probe_ok = False
        for ccy in currency_candidates:
            try:
                order = ig.place_order(
                    WorkingOrder(
                        id="",
                        type=probe_type,
                        side=probe_side,
                        level=float(probe_level),
                        size=size,
                        purpose=OrderPurpose.ENTRY,
                    ),
                    currency=ccy,
                )
                placed.append(order)
                working_currency = ccy
                probe_ok = True
                lines.append(
                    f"ACCEPTED {probe_label} @ {order.level:.2f} · currency={ccy} "
                    f"· dealId={order.deal_id or '—'} · ref={order.client_ref or '—'}"
                )
                break
            except IgAuthError as exc:
                lines.append(f"REJECTED {probe_label} with currency={ccy}: {exc}")
        if not probe_ok:
            lines.append("")
            lines.append(
                "Could not place a working order. If account is CFD, set Epic to the "
                "France 40 CFD instrument (not …DAILY.IP). Save the connector, then retry."
            )
            lines.append("")
            lines.append(context)
            return ConnectorTestResult(ok=False, message="\n".join(lines), error=context)

        accepted = 1
        rejected = 0
        for otype, side, level, label in specs[1:]:
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
                )
                placed.append(order)
                accepted += 1
                lines.append(
                    f"ACCEPTED {label} @ {order.level:.2f} · currency={working_currency} "
                    f"· dealId={order.deal_id or '—'} · ref={order.client_ref or '—'}"
                )
            except IgAuthError as exc:
                rejected += 1
                lines.append(f"REJECTED {label} @ {level:.2f}: {exc}")
        lines.append("")
        lines.append(f"Using currency={working_currency} for this test.")
        lines.append("")
        open_orders = ig.list_working_orders()
        lines.append(f"Open working orders on account now: {len(open_orders)}")
        for row in open_orders:
            lines.append(
                f"  · dealId={row.get('dealId') or '—'} "
                f"{row.get('direction') or ''} {row.get('orderType') or row.get('type') or ''} "
                f"@ {row.get('orderLevel') or row.get('level') or '—'} "
                f"epic={row.get('epic') or '—'}"
            )
        if not open_orders:
            lines.append("  (none)")
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
            f"still_open={len(remaining)}"
        )
        lines.append("")
        lines.append(context)
        ok = accepted > 0 and rejected == 0 and cancel_fail == 0 and len(remaining) == 0
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
