from __future__ import annotations

from unittest.mock import MagicMock

from chatbot.application.customer_access_gate import (
    CustomerAccessGate,
    CustomerContext,
    format_context,
    parse_session_identity,
)


def test_parse_session_identity_whatsapp() -> None:
    email, phone = parse_session_identity("whatsapp:33612345678")
    assert email is None
    assert phone == "33612345678"


def test_parse_session_identity_email() -> None:
    email, phone = parse_session_identity("email:Alice@Example.com")
    assert email == "alice@example.com"
    assert phone is None


def test_parse_session_identity_unknown_channel() -> None:
    email, phone = parse_session_identity("dashboard:42")
    assert email is None
    assert phone is None


def test_format_context_includes_orders_and_quotations() -> None:
    text = format_context(
        CustomerContext(
            customer_name="Alice Corp",
            orders=[{"name": "SO-1", "transaction_date": "2026-01-01", "status": "Open", "grand_total": 100}],
            quotations=[{"name": "QT-1", "transaction_date": "2026-02-01", "status": "Draft", "grand_total": 50}],
            source_label="ERPNext",
        )
    )
    assert "Alice Corp" in text
    assert "SO-1" in text
    assert "QT-1" in text


def test_format_context_includes_product_lines() -> None:
    text = format_context(
        CustomerContext(
            customer_name="Alice Corp",
            orders=[
                {
                    "name": "SINV-1",
                    "transaction_date": "2026-01-01",
                    "status": "Paid",
                    "grand_total": 100,
                    "items": [{"item_name": "Widget", "item_code": "W-1", "qty": 2, "rate": 50, "uom": "Nos"}],
                }
            ],
            quotations=[
                {
                    "name": "QT-1",
                    "transaction_date": "2026-02-01",
                    "status": "Open",
                    "grand_total": 50,
                    "items": [{"item_name": "Gadget", "item_code": "G-1", "qty": 1, "rate": 50, "uom": "Nos"}],
                }
            ],
            source_label="ERPNext",
        )
    )
    assert "Widget x2 Nos @50" in text
    assert "Gadget x1 Nos @50" in text


def test_format_context_includes_company_and_contact() -> None:
    text = format_context(
        CustomerContext(
            customer_name="Alice Corp",
            orders=[],
            quotations=[],
            source_label="ERPNext",
            company={
                "name": "Alice Corp",
                "customer_type": "Company",
                "email": "info@alice.example",
                "address": {
                    "line1": "1 Main St",
                    "city": "Port Louis",
                    "country": "Mauritius",
                },
            },
            contact={
                "full_name": "Alice Smith",
                "email": "alice@example.com",
                "mobile": "12345678",
                "designation": "Buyer",
            },
        )
    )
    assert "Company:" in text
    assert "Contact:" in text
    assert "customer_type: Company" in text
    assert "alice@example.com" in text
    assert "full_name: Alice Smith" in text
    assert "1 Main St, Port Louis, Mauritius" in text


def test_format_context_includes_current_prices() -> None:
    text = format_context(
        CustomerContext(
            customer_name="Alice Corp",
            orders=[
                {
                    "name": "SINV-1",
                    "transaction_date": "2026-01-01",
                    "status": "Paid",
                    "grand_total": 100,
                    "items": [{"item_name": "Widget", "item_code": "W-1", "qty": 2, "rate": 50, "uom": "Nos"}],
                }
            ],
            quotations=[],
            source_label="ERPNext",
            current_prices={
                "W-1": {
                    "current_rate": 55,
                    "currency": "MUR",
                    "uom": "Nos",
                    "source": "price_list",
                    "price_list": "Standard Selling",
                }
            },
        )
    )
    assert "Current list prices" in text
    assert "W-1: 55 MUR/Nos (Standard Selling)" in text
    assert "current list @55" in text


def test_format_context_omits_current_prices_when_all_zero() -> None:
    text = format_context(
        CustomerContext(
            customer_name="Alice Corp",
            orders=[],
            quotations=[],
            source_label="ERPNext",
            current_prices={"W-1": {"current_rate": 0, "uom": "Nos", "source": "price_list"}},
        )
    )
    assert "Current list prices" not in text


def test_gate_enrich_whatsapp_customer() -> None:
    client = MagicMock()
    client.find_customer.return_value = "Alice Corp"
    client.get_customer_profile.return_value = {"name": "Alice Corp"}
    client.get_matched_contact.return_value = {"full_name": "Alice", "mobile": "33612345678"}
    client.get_orders.return_value = [
        {
            "name": "SO-1",
            "transaction_date": "2026-01-01",
            "status": "Open",
            "grand_total": 10,
            "items": [{"item_code": "W-1", "item_name": "Widget", "qty": 1, "rate": 10}],
        }
    ]
    client.get_quotations.return_value = []
    client.get_current_item_prices.return_value = {"W-1": {"current_rate": 12, "source": "price_list"}}
    gate = CustomerAccessGate(
        client,
        {"fetch_orders": True, "fetch_quotations": True, "max_items": 5},
        source_label="ERPNext",
    )
    block = gate.enrich("whatsapp:33612345678")
    assert block is not None
    assert "Alice Corp" in block
    client.get_current_item_prices.assert_called_once()
    client.find_customer.assert_called_once_with(email=None, phone="33612345678")


def test_gate_enrich_skips_untrusted_session() -> None:
    client = MagicMock()
    gate = CustomerAccessGate(client, {}, source_label="ERPNext")
    assert gate.enrich("dashboard:1") is None
    client.find_customer.assert_not_called()
