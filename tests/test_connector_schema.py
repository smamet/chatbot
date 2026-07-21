from __future__ import annotations

from chatbot.domain.models.connector import ConnectorDirection, ConnectorType
from chatbot.domain.models.connector_schema import (
    CONNECTOR_ALLOWLIST_NONE,
    CONNECTOR_SCHEMAS,
    EmailOutboundProvider,
    all_connector_capability_keys,
    connector_capabilities_for_ui,
    filter_connector_schemas,
    fields_for,
    is_connector_allowed,
    normalize_allowed_connectors,
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
    assert "process_since" in in_keys
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
    for ctype in ("whatsapp", "messenger", "instagram", "email", "ig"):
        assert ctype in CONNECTOR_SCHEMAS
        assert CONNECTOR_SCHEMAS[ctype]


def test_ig_fields_are_bidirectional_only() -> None:
    fields = CONNECTOR_SCHEMAS[ConnectorType.IG.value]
    assert fields
    for field in fields:
        assert field.directions == (ConnectorDirection.BOTH.value,)
    both_keys = {f.key for f in fields_for(ConnectorType.IG.value, ConnectorDirection.BOTH.value)}
    assert "api_key" in both_keys
    assert "dry_run" in both_keys
    assert fields_for(ConnectorType.IG.value, ConnectorDirection.IN.value) == []
    assert fields_for(ConnectorType.IG.value, ConnectorDirection.OUT.value) == []


def test_ig_capability_is_both_not_in_out() -> None:
    keys = all_connector_capability_keys()
    assert "ig:both" in keys
    assert "ig:in" not in keys
    assert "ig:out" not in keys


def test_is_connector_allowed_empty_means_all() -> None:
    assert is_connector_allowed((), "whatsapp", "in") is True
    assert is_connector_allowed(None, "email", "out") is True
    assert is_connector_allowed((), "ig", "both") is True


def test_is_connector_allowed_partial_deny() -> None:
    allowed = ("whatsapp:in", "email:out")
    assert is_connector_allowed(allowed, "whatsapp", "in") is True
    assert is_connector_allowed(allowed, "whatsapp", "out") is False
    assert is_connector_allowed(allowed, "email", "out") is True
    assert is_connector_allowed(allowed, "email", "in") is False
    assert is_connector_allowed(allowed, "ig", "both") is False


def test_is_connector_allowed_ig_legacy_aliases() -> None:
    assert is_connector_allowed(("ig:out",), "ig", "both") is True
    assert is_connector_allowed(("ig:in",), "ig", "both") is True
    assert is_connector_allowed(("ig:both",), "ig", "in") is True
    assert is_connector_allowed(("ig:both",), "ig", "out") is True
    assert is_connector_allowed(("ig:both",), "ig", "both") is True
    assert is_connector_allowed(("whatsapp:in",), "ig", "both") is False


def test_is_connector_allowed_none_sentinel() -> None:
    assert is_connector_allowed((CONNECTOR_ALLOWLIST_NONE,), "whatsapp", "in") is False
    assert is_connector_allowed((CONNECTOR_ALLOWLIST_NONE,), "ig", "both") is False


def test_normalize_allowed_connectors_round_trips() -> None:
    all_keys = list(all_connector_capability_keys())
    assert normalize_allowed_connectors(all_keys) == ()
    assert normalize_allowed_connectors([]) == (CONNECTOR_ALLOWLIST_NONE,)
    assert normalize_allowed_connectors(["whatsapp:in", "email:out"]) == (
        "email:out",
        "whatsapp:in",
    )


def test_normalize_allowed_connectors_maps_legacy_ig() -> None:
    assert normalize_allowed_connectors(["ig:in", "whatsapp:out"]) == (
        "ig:both",
        "whatsapp:out",
    )
    assert normalize_allowed_connectors(["ig:out", "ig:in"]) == ("ig:both",)


def test_filter_connector_schemas_respects_allowlist() -> None:
    filtered = filter_connector_schemas(("whatsapp:in",))
    assert "whatsapp" in filtered
    assert "email" not in filtered
    assert "ig" not in filtered
    dirs = {d for field in filtered["whatsapp"] for d in field.directions}
    assert dirs == {"in"}


def test_filter_connector_schemas_ig_both() -> None:
    assert "ig" not in filter_connector_schemas(("whatsapp:in",))
    filtered = filter_connector_schemas(("ig:both",))
    assert "ig" in filtered
    dirs = {d for field in filtered["ig"] for d in field.directions}
    assert dirs == {"both"}
    # Legacy allowlist still exposes IG.
    legacy = filter_connector_schemas(("ig:out",))
    assert "ig" in legacy


def test_filter_connector_schemas_empty_allowlist_includes_ig() -> None:
    filtered = filter_connector_schemas(())
    assert "ig" in filtered


def test_connector_capabilities_ui_ig_label() -> None:
    rows = connector_capabilities_for_ui(())
    ig = next(r for r in rows if r["key"] == "ig:both")
    assert ig["label"] == "IG"
    assert ig["checked"] is True
    assert not any(r["key"] in ("ig:in", "ig:out") for r in rows)
