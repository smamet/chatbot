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
    epic = str(config.get("epic", "") or "IX.D.CAC.DAILY.IP").strip()
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
