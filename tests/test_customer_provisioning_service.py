from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chatbot.application.customer_access_gate import can_create_customer, can_create_quotation
from chatbot.application.customer_provisioning_service import (
    CustomerProvisioningError,
    create_erpnext_customer_for_test,
    ensure_erpnext_customer,
)


def test_permission_helpers_default_false() -> None:
    assert can_create_customer({}) is False
    assert can_create_quotation({}) is False
    assert can_create_customer({"allow_create_customer": "true"}) is True
    assert can_create_quotation({"allow_create_quotation": "on"}) is True


def test_ensure_erpnext_customer_returns_existing() -> None:
    client = MagicMock()
    client.find_customer.return_value = "Existing Corp"
    name = ensure_erpnext_customer(
        client,
        {"allow_create_customer": True},
        email="a@example.com",
        phone=None,
    )
    assert name == "Existing Corp"
    client.create_customer.assert_not_called()


def test_ensure_erpnext_customer_requires_permission() -> None:
    client = MagicMock()
    client.find_customer.return_value = None
    with pytest.raises(CustomerProvisioningError, match="disabled"):
        ensure_erpnext_customer(
            client,
            {"allow_create_customer": False},
            email="a@example.com",
            phone=None,
        )


def test_ensure_erpnext_customer_creates_when_allowed() -> None:
    client = MagicMock()
    client.find_customer.return_value = None
    client.create_customer.return_value = {"customer_name": "Alice Corp", "contact_name": "C-1"}
    name = ensure_erpnext_customer(
        client,
        {"allow_create_customer": True},
        email="alice@example.com",
        phone=None,
        customer_name="Alice Corp",
    )
    assert name == "Alice Corp"
    client.create_customer.assert_called_once()


def test_ensure_erpnext_customer_with_company() -> None:
    client = MagicMock()
    client.find_customer.return_value = None
    client.create_customer.return_value = {
        "customer_name": "Acme Corp",
        "contact_name": "C-1",
        "company_name": "Acme Corp",
    }
    name = ensure_erpnext_customer(
        client,
        {"allow_create_customer": True},
        email="samuel@example.com",
        phone=None,
        customer_name="Samuel MAMET",
        company_name="Acme Corp",
    )
    assert name == "Acme Corp"
    client.create_customer.assert_called_once_with(
        "Samuel MAMET",
        email="samuel@example.com",
        phone=None,
        company_name="Acme Corp",
        customer_group=None,
    )


def test_create_erpnext_customer_for_test_disabled() -> None:
    client = MagicMock()
    out = create_erpnext_customer_for_test(
        client,
        {"allow_create_customer": False},
        test_email="a@example.com",
        test_phone="",
    )
    assert out["ok"] is False
    assert out["error"] == "creation_disabled"


def test_create_erpnext_customer_for_test_existing() -> None:
    client = MagicMock()
    client.find_customer.return_value = "Bob SA"
    out = create_erpnext_customer_for_test(
        client,
        {"allow_create_customer": True},
        test_email="bob@example.com",
        test_phone="",
    )
    assert out["ok"] is True
    assert out["created"] is False
    assert out["customer"] == "Bob SA"
