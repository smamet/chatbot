from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from chatbot.adapters.erpnext.client import ErpNextClient, _phone_variants


def _config() -> dict:
    return {
        "url": "https://erp.example.com",
        "api_key": "key",
        "api_secret": "secret",
        "identity_email_field": "email_id",
        "identity_phone_field": "mobile_no",
    }


def test_phone_variants() -> None:
    variants = _phone_variants("+33612345678")
    assert "+33612345678" in variants
    assert "33612345678" in variants


def test_find_customer_by_email() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/Contact-1"):
            return {
                "data": {
                    "name": "Contact-1",
                    "links": [{"link_doctype": "Customer", "link_name": "Alice Corp"}],
                }
            }
        assert "Contact" in path
        filters = json.loads(params["filters"])  # type: ignore[index]
        assert filters == [["email_id", "=", "alice@example.com"]]
        return {"data": [{"name": "Contact-1"}]}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(email="alice@example.com") == "Alice Corp"


def test_find_customer_fetches_full_contact_when_list_omits_links() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/United%20Docks%20Business%20Park-United%20Docks%20Business%20Park"):
            return {
                "data": {
                    "name": "United Docks Business Park-United Docks Business Park",
                    "links": [
                        {
                            "link_doctype": "Customer",
                            "link_name": "United Docks Business Park",
                        }
                    ],
                }
            }
        filters = json.loads(params["filters"])  # type: ignore[index]
        assert filters == [["email_id", "=", "agoburdhun@uniteddocks.com"]]
        return {
            "data": [{"name": "United Docks Business Park-United Docks Business Park"}]
        }

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(email="agoburdhun@uniteddocks.com") == "United Docks Business Park"


def test_find_customer_by_phone() -> None:
    client = ErpNextClient(_config())
    calls: list[str] = []

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/Contact-2"):
            return {
                "data": {
                    "name": "Contact-2",
                    "links": [{"link_doctype": "Customer", "link_name": "Bob SA"}],
                }
            }
        filters = json.loads(params["filters"])  # type: ignore[index]
        calls.append(filters[0][2])
        if filters[0][2] == "+33612345678":
            return {"data": [{"name": "Contact-2"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(phone="33612345678") == "Bob SA"
    assert "+33612345678" in calls or "33612345678" in calls


def test_get_sales_invoices_and_quotations_with_line_items() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Sales Invoice/SINV-001"):
            return {
                "data": {
                    "name": "SINV-001",
                    "items": [
                        {
                            "item_code": "ITEM-1",
                            "item_name": "Widget",
                            "qty": 2,
                            "rate": 10,
                            "amount": 20,
                            "uom": "Nos",
                        }
                    ],
                }
            }
        if path.endswith("/Quotation/QT-001"):
            return {
                "data": {
                    "name": "QT-001",
                    "items": [
                        {
                            "item_code": "ITEM-2",
                            "item_name": "Gadget",
                            "qty": 1,
                            "rate": 5,
                            "amount": 5,
                            "uom": "Nos",
                        }
                    ],
                }
            }
        if "Sales Invoice" in path:
            return {
                "data": [
                    {
                        "name": "SINV-001",
                        "posting_date": "2026-01-01",
                        "status": "Paid",
                        "grand_total": 20,
                    }
                ]
            }
        if "Quotation" in path:
            return {
                "data": [
                    {
                        "name": "QT-001",
                        "transaction_date": "2026-02-01",
                        "status": "Open",
                        "grand_total": 5,
                    }
                ]
            }
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        orders = client.get_orders("Alice Corp", 5)
        quotes = client.get_quotations("Alice Corp", 5)
    assert orders[0]["name"] == "SINV-001"
    assert orders[0]["transaction_date"] == "2026-01-01"
    assert orders[0]["items"][0]["item_name"] == "Widget"
    assert quotes[0]["name"] == "QT-001"
    assert quotes[0]["items"][0]["item_name"] == "Gadget"


def test_get_sales_orders_list_only() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if "Sales Order" in path:
            return {"data": [{"name": "SO-001", "status": "To Deliver"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        orders = client.get_sales_orders("Alice Corp", 5)
    assert orders[0]["name"] == "SO-001"


def test_get_customer_profile_and_contact() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Customer/Alice%20Corp"):
            return {
                "data": {
                    "name": "Alice Corp",
                    "customer_name": "Alice Corp",
                    "customer_type": "Company",
                    "customer_group": "Commercial",
                    "territory": "All Territories",
                    "email_id": "info@alice.example",
                    "mobile_no": "1111",
                    "customer_primary_contact": "Alice Smith",
                }
            }
        if path.endswith("/Contact/Alice%20Smith"):
            return {
                "data": {
                    "name": "Alice Smith",
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "email_id": "alice@example.com",
                    "mobile_no": "2222",
                    "designation": "Buyer",
                }
            }
        if path.endswith("/Address/Alice%20Corp-Billing"):
            return {
                "data": {
                    "address_title": "Billing",
                    "address_line1": "1 Main St",
                    "city": "Port Louis",
                    "country": "Mauritius",
                }
            }
        if "Address" in path:
            return {"data": [{"name": "Alice Corp-Billing"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        profile = client.get_customer_profile("Alice Corp")
        contact = client.get_matched_contact(
            email="alice@example.com",
            phone=None,
            customer="Alice Corp",
        )
    assert profile["name"] == "Alice Corp"
    assert profile["customer_type"] == "Company"
    assert profile["address"]["city"] == "Port Louis"
    assert contact is not None
    assert contact["name"] == "Alice Smith"
    assert contact["email"] == "alice@example.com"


def test_find_customer_returns_none_on_http_error() -> None:
    client = ErpNextClient(_config())
    with patch.object(client, "_get", return_value={}):
        assert client.find_customer(email="missing@example.com") is None
