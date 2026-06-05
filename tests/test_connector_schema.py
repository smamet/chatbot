from __future__ import annotations

from chatbot.domain.models.connector import ConnectorDirection, ConnectorType
from chatbot.domain.models.connector_schema import (
    CONNECTOR_SCHEMAS,
    EmailOutboundProvider,
    fields_for,
    secret_connector_keys,
)


def test_whatsapp_out_fields_exclude_verify_token() -> None:
    fields = fields_for(ConnectorType.WHATSAPP.value, ConnectorDirection.OUT.value)
    keys = {field.key for field in fields}
    assert "phone_number_id" in keys
    assert "access_token" in keys
    assert "verify_token" not in keys


def test_email_in_vs_out_fields() -> None:
    in_keys = {f.key for f in fields_for(ConnectorType.EMAIL.value, ConnectorDirection.IN.value)}
    out_keys = {
        f.key
        for f in fields_for(
            ConnectorType.EMAIL.value,
            ConnectorDirection.OUT.value,
            outbound_provider=EmailOutboundProvider.SMTP.value,
        )
    }
    assert "imap_host" in in_keys
    assert "username" in in_keys
    assert "smtp_host" not in in_keys
    assert "outbound_provider" in out_keys
    assert "smtp_host" in out_keys
    assert "from_addr" in out_keys
    assert "username" not in out_keys


def test_secret_keys_include_passwords_and_tokens() -> None:
    secrets = secret_connector_keys()
    assert "password" in secrets
    assert "access_token" in secrets
    assert "mailjet_api_key" in secrets
    assert "mailgun_api_key" in secrets
    assert "phone_number_id" not in secrets


def test_all_dashboard_types_have_schemas() -> None:
    for ctype in ("whatsapp", "messenger", "instagram", "email"):
        assert ctype in CONNECTOR_SCHEMAS
        assert CONNECTOR_SCHEMAS[ctype]
