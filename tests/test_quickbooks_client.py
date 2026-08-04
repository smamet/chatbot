from __future__ import annotations

from unittest.mock import patch

from evenor.adapters.quickbooks.client import QuickBooksClient


def _config() -> dict:
    return {
        "environment": "sandbox",
        "client_id": "cid",
        "client_secret": "sec",
        "realm_id": "12345",
        "access_token": "access",
        "refresh_token": "refresh",
        "token_expires_at": 9999999999,
    }


def test_ping_and_find_customer() -> None:
    client = QuickBooksClient(_config())

    def fake_query(sql: str) -> list[dict]:
        if "FROM Customer MAXRESULTS 1" in sql:
            return [{"Id": "1"}]
        if "PrimaryEmailAddr" in sql:
            return [{"Id": "42", "DisplayName": "Alice Corp"}]
        return []

    with patch.object(client, "_query", side_effect=fake_query):
        client.ping()
        assert client.find_customer(email="alice@example.com") == "Alice Corp"


def test_get_orders_uses_customer_id() -> None:
    client = QuickBooksClient(_config())
    queries: list[str] = []

    def fake_query(sql: str) -> list[dict]:
        queries.append(sql)
        if "PrimaryEmailAddr" in sql:
            return [{"Id": "42", "DisplayName": "Alice Corp"}]
        if "FROM Invoice" in sql:
            return [{"DocNumber": "INV-1", "TxnDate": "2026-01-01", "TotalAmt": 10, "EmailStatus": "Sent"}]
        return []

    with patch.object(client, "_query", side_effect=fake_query):
        client.find_customer(email="alice@example.com")
        orders = client.get_orders("Alice Corp", 3)
    assert orders[0]["name"] == "INV-1"
    assert any("CustomerRef = '42'" in q for q in queries)
