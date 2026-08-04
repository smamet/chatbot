from __future__ import annotations

from unittest.mock import MagicMock, patch

from evenor.adapters.erpnext.client import ErpNextClient


def test_search_items_by_name() -> None:
    client = ErpNextClient({"url": "https://erp.example.com", "api_key": "k", "api_secret": "s"})
    with patch.object(client, "_list_resource", return_value=[{"item_code": "A", "item_name": "Alpha"}]):
        rows = client.search_items("alpha", limit=5)
    assert rows[0]["item_code"] == "A"


def test_create_quotation_posts_payload() -> None:
    client = ErpNextClient({"url": "https://erp.example.com", "api_key": "k", "api_secret": "s"})
    with patch.object(client, "_post", return_value={"data": {"name": "QTN-0001"}}) as mock_post:
        out = client.create_quotation(
            "CUST-1",
            [{"item_code": "SKU", "qty": 2, "rate": 10}],
            notes="Test",
        )
    assert out["name"] == "QTN-0001"
    mock_post.assert_called_once()
