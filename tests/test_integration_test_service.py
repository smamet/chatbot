from __future__ import annotations

from unittest.mock import MagicMock

from chatbot.application.integration_test_service import run_integration_test


def test_run_integration_test_missing_identity() -> None:
    result = run_integration_test("erpnext", {"url": "https://erp.test"}, test_email="", test_phone="")
    assert result.ok is False
    assert result.error == "missing_test_identity"


def test_run_integration_test_success() -> None:
    client = MagicMock()
    client.ping.return_value = None
    client.find_customer.return_value = "Alice Corp"
    client.get_customer_profile.return_value = {
        "name": "Alice Corp",
        "customer_type": "Company",
    }
    client.get_matched_contact.return_value = {
        "name": "Alice Smith",
        "email": "alice@example.com",
    }
    client.get_orders.return_value = [{"name": "SO-1", "transaction_date": "2026-01-01", "status": "Open", "grand_total": 1}]
    client.get_quotations.return_value = []

    from chatbot.application import integration_test_service as mod

    original = mod._gate_for_type

    def fake_gate(integration_type: str, config: dict):
        _ = integration_type, config
        from chatbot.application.customer_access_gate import CustomerAccessGate

        return CustomerAccessGate(
            client,
            {"fetch_orders": True, "fetch_quotations": True, "max_items": 5},
            source_label="ERPNext",
        )

    mod._gate_for_type = fake_gate
    try:
        result = run_integration_test(
            "erpnext",
            {"url": "https://erp.test", "api_key": "k", "api_secret": "s"},
            test_email="alice@example.com",
        )
    finally:
        mod._gate_for_type = original

    assert result.ok is True
    assert result.customer == "Alice Corp"
    assert "Alice Corp" in (result.preview or "")
    assert "Company:" in (result.preview or "")
    assert "Contact:" in (result.preview or "")
