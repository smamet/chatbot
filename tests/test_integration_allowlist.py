from __future__ import annotations

from chatbot.domain.models.integration_schema import (
    INTEGRATION_ALLOWLIST_NONE,
    all_integration_types,
    filter_integration_types,
    is_integration_allowed,
    normalize_allowed_integrations,
)


def test_is_integration_allowed_empty_means_all() -> None:
    assert is_integration_allowed((), "erpnext") is True
    assert is_integration_allowed(None, "quickbooks") is True


def test_is_integration_allowed_partial_deny() -> None:
    allowed = ("erpnext", "cac40_backtest")
    assert is_integration_allowed(allowed, "erpnext") is True
    assert is_integration_allowed(allowed, "quickbooks") is False
    assert is_integration_allowed(allowed, "cac40_backtest") is True


def test_is_integration_allowed_none_sentinel() -> None:
    assert is_integration_allowed((INTEGRATION_ALLOWLIST_NONE,), "erpnext") is False


def test_normalize_allowed_integrations_round_trips() -> None:
    all_keys = list(all_integration_types())
    assert normalize_allowed_integrations(all_keys) == ()
    assert normalize_allowed_integrations([]) == (INTEGRATION_ALLOWLIST_NONE,)
    assert normalize_allowed_integrations(["erpnext", "quickbooks"]) == (
        "erpnext",
        "quickbooks",
    )


def test_filter_integration_types_respects_allowlist() -> None:
    assert filter_integration_types(("erpnext",)) == ["erpnext"]
    assert "quickbooks" not in filter_integration_types(("erpnext",))
