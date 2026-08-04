from __future__ import annotations

from typing import Any

from evenor.domain.models.connector import ConnectorDirection
from evenor.domain.models.connector_schema import EmailAuthType
from evenor.domain.models.mail_connection import MailConnection, MailConnectionProvider

_PROVIDER_PRESETS: dict[MailConnectionProvider, dict[str, Any]] = {
    MailConnectionProvider.MICROSOFT_OAUTH: {
        "auth_type": EmailAuthType.MICROSOFT_OAUTH.value,
        "imap_host": "outlook.office365.com",
        "imap_port": "993",
        "imap_use_ssl": True,
        "smtp_host": "smtp.office365.com",
        "smtp_port": "587",
        "smtp_use_tls": True,
    },
    MailConnectionProvider.GOOGLE_OAUTH: {
        "auth_type": EmailAuthType.GOOGLE_OAUTH.value,
        "imap_host": "imap.gmail.com",
        "imap_port": "993",
        "imap_use_ssl": True,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": "587",
        "smtp_use_tls": True,
    },
}

_OAUTH_CONFIG_KEYS = (
    "oauth_refresh_token",
    "oauth_access_token",
    "oauth_token_expires_at",
    "microsoft_client_id",
    "microsoft_client_secret",
    "google_client_id",
    "google_client_secret",
)


def provider_auth_type(provider: MailConnectionProvider) -> str:
    return _PROVIDER_PRESETS[provider]["auth_type"]


def build_runtime_mail_config(connection: MailConnection, *, direction: str) -> dict[str, Any]:
    preset = _PROVIDER_PRESETS[connection.provider]
    oauth_fields = {key: connection.config.get(key) for key in _OAUTH_CONFIG_KEYS if key in connection.config}
    mailbox = connection.mailbox_email.strip()
    if direction == ConnectorDirection.IN.value:
        return {
            "auth_type": preset["auth_type"],
            "imap_host": preset["imap_host"],
            "imap_port": preset["imap_port"],
            "imap_use_ssl": preset["imap_use_ssl"],
            "username": mailbox,
            **oauth_fields,
        }
    return {
        "auth_type": preset["auth_type"],
        "smtp_host": preset["smtp_host"],
        "smtp_port": preset["smtp_port"],
        "smtp_use_tls": preset["smtp_use_tls"],
        "smtp_username": mailbox,
        **oauth_fields,
    }
