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


def test_gate_enrich_whatsapp_customer() -> None:
    client = MagicMock()
    client.find_customer.return_value = "Alice Corp"
    client.get_orders.return_value = [{"name": "SO-1", "transaction_date": "2026-01-01", "status": "Open", "grand_total": 10}]
    client.get_quotations.return_value = []
    gate = CustomerAccessGate(
        client,
        {"fetch_orders": True, "fetch_quotations": True, "max_items": 5},
        source_label="ERPNext",
    )
    block = gate.enrich("whatsapp:33612345678")
    assert block is not None
    assert "Alice Corp" in block
    client.find_customer.assert_called_once_with(email=None, phone="33612345678")


def test_gate_enrich_skips_untrusted_session() -> None:
    client = MagicMock()
    gate = CustomerAccessGate(client, {}, source_label="ERPNext")
    assert gate.enrich("dashboard:1") is None
    client.find_customer.assert_not_called()
