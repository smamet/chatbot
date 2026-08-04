from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evenor.adapters.google import oauth as google_oauth
from evenor.adapters.microsoft import oauth as microsoft_oauth
from evenor.domain.models.connector_schema import (
    EmailAuthType,
    is_oauth_auth_type,
    resolve_email_auth_type,
)
from evenor.domain.models.mail_connection import MailConnection, MailConnectionProvider

if TYPE_CHECKING:
    from evenor.config.settings import Settings


class MailOAuthError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OAuthAccessResult:
    access_token: str
    updated_config: dict[str, Any] | None


def platform_microsoft_mail_oauth_configured(settings: Settings) -> bool:
    return bool(
        settings.microsoft_mail_client_id.strip()
        and settings.microsoft_mail_client_secret.strip()
    )


def platform_google_mail_oauth_configured(settings: Settings) -> bool:
    return bool(
        settings.google_mail_client_id.strip() and settings.google_mail_client_secret.strip()
    )


def platform_oauth_credentials_for_provider(
    provider: MailConnectionProvider | str,
    settings: Settings,
) -> tuple[str, str] | None:
    provider_value = provider.value if isinstance(provider, MailConnectionProvider) else str(provider)
    if provider_value == MailConnectionProvider.MICROSOFT_OAUTH.value:
        client_id = settings.microsoft_mail_client_id.strip()
        client_secret = settings.microsoft_mail_client_secret.strip()
    elif provider_value == MailConnectionProvider.GOOGLE_OAUTH.value:
        client_id = settings.google_mail_client_id.strip()
        client_secret = settings.google_mail_client_secret.strip()
    else:
        return None
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


def resolve_mail_oauth_credentials(
    connection: MailConnection,
    settings: Settings,
) -> tuple[str, str]:
    """Return (client_id, client_secret): platform env creds override per-connection config."""
    platform = platform_oauth_credentials_for_provider(connection.provider, settings)
    if platform is not None:
        return platform
    cfg = dict(connection.config)
    cfg["auth_type"] = _provider_auth_type(connection.provider)
    return oauth_client_credentials(cfg)


def apply_platform_oauth_credentials_to_config(
    config: dict,
    settings: Settings | None,
) -> dict:
    if settings is None:
        return config
    auth_type = resolve_email_auth_type(config)
    if auth_type == EmailAuthType.MICROSOFT_OAUTH.value:
        creds = platform_oauth_credentials_for_provider(
            MailConnectionProvider.MICROSOFT_OAUTH, settings
        )
        if creds is None:
            return config
        client_id, client_secret = creds
        merged = dict(config)
        merged["microsoft_client_id"] = client_id
        merged["microsoft_client_secret"] = client_secret
        return merged
    if auth_type == EmailAuthType.GOOGLE_OAUTH.value:
        creds = platform_oauth_credentials_for_provider(
            MailConnectionProvider.GOOGLE_OAUTH, settings
        )
        if creds is None:
            return config
        client_id, client_secret = creds
        merged = dict(config)
        merged["google_client_id"] = client_id
        merged["google_client_secret"] = client_secret
        return merged
    return config


def _provider_auth_type(provider: MailConnectionProvider) -> str:
    if provider == MailConnectionProvider.MICROSOFT_OAUTH:
        return EmailAuthType.MICROSOFT_OAUTH.value
    return EmailAuthType.GOOGLE_OAUTH.value


def is_oauth_connected(config: dict) -> bool:
    auth_type = resolve_email_auth_type(config)
    if not is_oauth_auth_type(auth_type):
        return False
    return has_oauth_refresh_token(config)


def has_oauth_refresh_token(config: dict) -> bool:
    return bool(str(config.get("oauth_refresh_token", "")).strip())


def oauth_client_credentials(config: dict) -> tuple[str, str]:
    auth_type = resolve_email_auth_type(config)
    if auth_type == EmailAuthType.MICROSOFT_OAUTH.value or config.get("microsoft_client_id"):
        client_id = str(config.get("microsoft_client_id", "")).strip()
        client_secret = str(config.get("microsoft_client_secret", "")).strip()
    elif auth_type == EmailAuthType.GOOGLE_OAUTH.value or config.get("google_client_id"):
        client_id = str(config.get("google_client_id", "")).strip()
        client_secret = str(config.get("google_client_secret", "")).strip()
    else:
        raise MailOAuthError("Connector is not configured for OAuth")
    if not client_id or not client_secret:
        raise MailOAuthError("OAuth client ID and secret are required")
    return client_id, client_secret


def _token_still_valid(config: dict) -> bool:
    expires_raw = config.get("oauth_token_expires_at")
    access = str(config.get("oauth_access_token", "")).strip()
    if not access or expires_raw is None or str(expires_raw).strip() == "":
        return False
    try:
        expires_at = int(expires_raw)
    except (TypeError, ValueError):
        return False
    return expires_at > int(time.time())


def _refresh_tokens(
    config: dict,
    *,
    direction: str,
    settings: Settings | None = None,
) -> OAuthAccessResult:
    auth_type = resolve_email_auth_type(config)
    refresh_token = str(config.get("oauth_refresh_token", "")).strip()
    if not refresh_token:
        raise MailOAuthError("Mailbox is not connected — complete OAuth authorization first")
    effective = apply_platform_oauth_credentials_to_config(config, settings)
    client_id, client_secret = oauth_client_credentials(effective)
    try:
        if auth_type == EmailAuthType.MICROSOFT_OAUTH.value:
            tokens = microsoft_oauth.refresh_access_token(
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                scopes=microsoft_oauth.scopes_for_connection(),
            )
        else:
            tokens = google_oauth.refresh_access_token(
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
            )
    except Exception as exc:
        raise MailOAuthError(f"OAuth token refresh failed: {exc}") from exc

    updated = dict(config)
    updated["oauth_refresh_token"] = tokens.refresh_token
    updated["oauth_access_token"] = tokens.access_token
    updated["oauth_token_expires_at"] = tokens.expires_at
    updated["_resolved_access_token"] = tokens.access_token
    return OAuthAccessResult(access_token=tokens.access_token, updated_config=updated)


def get_oauth_access_token(
    config: dict,
    *,
    direction: str,
    settings: Settings | None = None,
    force_refresh: bool = False,
) -> OAuthAccessResult:
    auth_type = resolve_email_auth_type(config)
    if not is_oauth_auth_type(auth_type):
        raise MailOAuthError("Connector is not configured for OAuth")
    if not force_refresh and _token_still_valid(config):
        access = str(config.get("oauth_access_token", "")).strip()
        mail_cfg = dict(config)
        mail_cfg["_resolved_access_token"] = access
        return OAuthAccessResult(access_token=access, updated_config=None)
    return _refresh_tokens(config, direction=direction, settings=settings)


def prepare_oauth_mail_config(
    config: dict,
    *,
    direction: str,
    settings: Settings | None = None,
    force_refresh: bool = False,
) -> tuple[dict, dict | None]:
    """Resolve OAuth access token and return config ready for IMAP/SMTP adapters."""
    auth_type = resolve_email_auth_type(config)
    if not is_oauth_auth_type(auth_type):
        return config, None
    result = get_oauth_access_token(
        config,
        direction=direction,
        settings=settings,
        force_refresh=force_refresh,
    )
    if result.updated_config is not None:
        return result.updated_config, result.updated_config
    mail_cfg = dict(config)
    mail_cfg["_resolved_access_token"] = result.access_token
    return mail_cfg, None


def apply_oauth_tokens_to_config(config: dict, tokens: Any) -> dict:
    updated = dict(config)
    updated["oauth_refresh_token"] = tokens.refresh_token
    updated["oauth_access_token"] = tokens.access_token
    updated["oauth_token_expires_at"] = tokens.expires_at
    return updated
