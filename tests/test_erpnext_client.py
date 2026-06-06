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
        assert "Contact" in path
        filters = json.loads(params["filters"])  # type: ignore[index]
        assert filters == [["email_id", "=", "alice@example.com"]]
        return {
            "data": [
                {
                    "name": "Contact-1",
                    "links": [{"link_doctype": "Customer", "link_name": "Alice Corp"}],
                }
            ]
        }

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(email="alice@example.com") == "Alice Corp"


def test_find_customer_by_phone() -> None:
    client = ErpNextClient(_config())
    calls: list[str] = []

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        filters = json.loads(params["filters"])  # type: ignore[index]
        calls.append(filters[0][2])
        if filters[0][2] == "+33612345678":
            return {
                "data": [
                    {
                        "name": "Contact-2",
                        "links": [{"link_doctype": "Customer", "link_name": "Bob SA"}],
                    }
                ]
            }
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(phone="33612345678") == "Bob SA"
    assert "+33612345678" in calls or "33612345678" in calls


def test_get_sales_orders_and_quotations() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if "Sales Order" in path:
            return {"data": [{"name": "SO-001", "status": "To Deliver"}]}
        if "Quotation" in path:
            return {"data": [{"name": "QT-001", "status": "Open"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        orders = client.get_sales_orders("Alice Corp", 5)
        quotes = client.get_quotations("Alice Corp", 5)
    assert orders[0]["name"] == "SO-001"
    assert quotes[0]["name"] == "QT-001"


def test_find_customer_returns_none_on_http_error() -> None:
    client = ErpNextClient(_config())
    with patch.object(client, "_get", return_value={}):
        assert client.find_customer(email="missing@example.com") is None
