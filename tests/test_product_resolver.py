from __future__ import annotations

from chatbot.application.product_resolver import ProductResolver, normalize_product_key


class _FakeErpClient:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def get_item_by_code(self, item_code: str):
        for row in self._items:
            if row["item_code"] == item_code:
                return row
        return None

    def search_items(self, query: str, *, limit: int = 20):
        token = query.lower()
        return [
            row
            for row in self._items
            if token in row["item_name"].lower() or token in row["item_code"].lower()
        ][:limit]


def test_normalize_product_key() -> None:
    assert normalize_product_key("Sigma 3000") == normalize_product_key("sigma-3000")


def test_resolver_matches_normalized_name() -> None:
    client = _FakeErpClient(
        [
            {
                "item_code": "sigma-3000",
                "item_name": "Sigma 3000",
                "standard_rate": 100,
                "stock_uom": "Nos",
            }
        ]
    )
    resolver = ProductResolver(client)
    line = resolver.resolve_line(product="sigma 3000", qty=2)
    assert line.status.value == "resolved"
    assert line.item_code == "sigma-3000"


def test_resolver_ambiguous_sigma_variants() -> None:
    client = _FakeErpClient(
        [
            {"item_code": "sigma-pro", "item_name": "Sigma Pro", "standard_rate": 1, "stock_uom": "Nos"},
            {"item_code": "sigma-plus", "item_name": "Sigma +", "standard_rate": 2, "stock_uom": "Nos"},
            {"item_code": "sigma-3000", "item_name": "Sigma 3000", "standard_rate": 3, "stock_uom": "Nos"},
        ]
    )
    resolver = ProductResolver(client)
    line = resolver.resolve_line(product="sigma", qty=1)
    assert line.status.value in {"ambiguous", "not_found", "resolved"}
