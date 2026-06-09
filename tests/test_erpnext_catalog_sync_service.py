from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_paths import safe_catalog_filename, tenant_catalog_dir
from chatbot.application.erpnext_catalog_sync_service import (
    catalog_price_list,
    catalog_sync_due,
    catalog_sync_enabled,
    render_item_markdown,
    sync_catalog_files,
    sync_erpnext_catalog_for_tenant,
)
from chatbot.application.sync_service import IngestSyncService
from tests.conftest import TestSettings as SettingsForTests


def test_list_resource_paginated_uses_limit_start() -> None:
    client = ErpNextClient(
        {
            "url": "https://erp.example.com",
            "api_key": "key",
            "api_secret": "secret",
        }
    )
    calls: list[int] = []

    def fake_list(
        doctype: str,
        *,
        filters: list[list[str]],
        fields: list[str],
        limit: int,
        order_by: str = "modified desc",
        limit_start: int = 0,
    ) -> list[dict]:
        calls.append(limit_start)
        if limit_start == 0:
            return [{"item_code": f"ITEM-{i}"} for i in range(500)]
        if limit_start == 500:
            return [{"item_code": "ITEM-500"}]
        return []

    with patch.object(client, "_list_resource", side_effect=fake_list):
        rows = client._list_resource_paginated(
            "Item",
            filters=[["disabled", "=", 0]],
            fields=["item_code"],
            page_length=500,
        )

    assert len(rows) == 501
    assert calls == [0, 500]


def test_fetch_stock_totals_aggregates_bins() -> None:
    client = ErpNextClient(
        {
            "url": "https://erp.example.com",
            "api_key": "key",
            "api_secret": "secret",
        }
    )
    with patch.object(
        client,
        "_list_resource_paginated",
        return_value=[
            {"item_code": "A", "actual_qty": 3},
            {"item_code": "A", "actual_qty": 2},
            {"item_code": "B", "actual_qty": 1.5},
        ],
    ):
        totals = client.fetch_stock_totals()
    assert totals == {"A": 5.0, "B": 1.5}


def test_list_catalog_items_skips_empty_codes() -> None:
    client = ErpNextClient(
        {
            "url": "https://erp.example.com",
            "api_key": "key",
            "api_secret": "secret",
        }
    )
    with patch.object(
        client,
        "_list_resource_paginated",
        return_value=[{"item_code": "OK"}, {"item_code": ""}, {"item_code": "  "}],
    ):
        rows = client.list_catalog_items()
    assert len(rows) == 1
    assert rows[0]["item_code"] == "OK"


def test_fetch_price_list_rates_builds_dict_and_skips_zero() -> None:
    client = ErpNextClient(
        {
            "url": "https://erp.example.com",
            "api_key": "key",
            "api_secret": "secret",
        }
    )
    captured: list[list[list[str]]] = []

    def fake_paginated(
        doctype: str,
        *,
        filters: list[list[str]],
        fields: list[str],
        page_length: int = 500,
        order_by: str = "modified asc",
    ) -> list[dict]:
        captured.append(filters)
        return [
            {
                "item_code": "8.2.7",
                "price_list_rate": 34176,
                "currency": "MUR",
                "uom": "Nos",
            },
            {"item_code": "FREE", "price_list_rate": 0, "currency": "MUR"},
            {
                "item_code": "SW002",
                "price_list_rate": 130,
                "currency": "MUR",
            },
        ]

    with patch.object(client, "_list_resource_paginated", side_effect=fake_paginated):
        prices = client.fetch_price_list_rates("Standard Selling")

    assert captured == [[["price_list", "=", "Standard Selling"]]]
    assert prices["8.2.7"]["rate"] == 34176.0
    assert prices["8.2.7"]["currency"] == "MUR"
    assert prices["8.2.7"]["price_list"] == "Standard Selling"
    assert "FREE" not in prices
    assert prices["SW002"]["rate"] == 130.0


def test_fetch_price_list_rates_empty_name() -> None:
    client = ErpNextClient(
        {
            "url": "https://erp.example.com",
            "api_key": "key",
            "api_secret": "secret",
        }
    )
    assert client.fetch_price_list_rates("") == {}
    assert client.fetch_price_list_rates("   ") == {}


def test_catalog_price_list_defaults_and_empty() -> None:
    assert catalog_price_list({}) == "Standard Selling"
    assert catalog_price_list({"catalog_price_list": "Custom"}) == "Custom"
    assert catalog_price_list({"catalog_price_list": ""}) == ""
    assert catalog_price_list({"catalog_price_list": "  "}) == ""


def test_render_item_markdown_uses_price_list_over_zero_standard_rate() -> None:
    md = render_item_markdown(
        {
            "item_code": "8.2.7",
            "item_name": "Keyboard",
            "standard_rate": 0,
            "stock_uom": "Nos",
        },
        stock_qty=None,
        sync_date="2026-06-09 12:00 UTC",
        include_stock=False,
        price_entry={
            "rate": 34176.0,
            "currency": "MUR",
            "price_list": "Standard Selling",
        },
    )
    assert "Price: 34176 MUR (Standard Selling)" in md
    assert "Standard rate" not in md
    assert "0.0" not in md


def test_render_item_markdown_falls_back_to_standard_rate() -> None:
    md = render_item_markdown(
        {
            "item_code": "W-1",
            "item_name": "Widget",
            "standard_rate": 10,
            "stock_uom": "Nos",
        },
        stock_qty=7,
        sync_date="2026-06-09 12:00 UTC",
        include_stock=True,
    )
    assert "Price: 10" in md
    assert "Standard rate" not in md


def test_render_item_markdown_not_available_when_no_price() -> None:
    md = render_item_markdown(
        {
            "item_code": "X",
            "item_name": "Unknown",
            "standard_rate": 0,
            "stock_uom": "Nos",
        },
        stock_qty=None,
        sync_date="2026-06-09 12:00 UTC",
        include_stock=False,
    )
    assert "Price: not available" in md
    assert "0.0" not in md


def test_render_item_markdown_includes_stock_and_sync_date() -> None:
    md = render_item_markdown(
        {
            "item_code": "W-1",
            "item_name": "Widget",
            "item_group": "Products",
            "standard_rate": 10,
            "stock_uom": "Nos",
            "description": "A widget",
        },
        stock_qty=7,
        sync_date="2026-06-09 12:00 UTC",
        include_stock=True,
    )
    assert "# Widget" in md
    assert "Item code: W-1" in md
    assert "Price: 10" in md
    assert "Total stock: 7 Nos" in md
    assert "Stock/price as of: 2026-06-09 12:00 UTC" in md
    assert "## Description" in md


def test_safe_catalog_filename_replaces_slashes() -> None:
    assert safe_catalog_filename("A/B") == "A-B"


def test_sync_catalog_files_writes_prunes_and_skips_unchanged(
    test_settings: SettingsForTests,
) -> None:
    slug = "catalog-bot"
    items = [
        {
            "item_code": "KEEP",
            "item_name": "Keep me",
            "standard_rate": 1,
            "stock_uom": "Nos",
        },
        {
            "item_code": "NEW",
            "item_name": "New item",
            "standard_rate": 2,
            "stock_uom": "Nos",
        },
    ]
    root = tenant_catalog_dir(test_settings, slug)
    stale = root / "OLD.md"
    stale.write_text("# old\n", encoding="utf-8")

    written1, removed1, _ = sync_catalog_files(
        test_settings,
        slug,
        items,
        {"KEEP": 4, "NEW": 1},
        include_stock=True,
        sync_date="2026-06-09 12:00 UTC",
    )
    assert written1 == 2
    assert removed1 == 1
    assert not stale.exists()
    assert (root / "KEEP.md").is_file()
    assert (root / "NEW.md").is_file()

    written2, removed2, _ = sync_catalog_files(
        test_settings,
        slug,
        items,
        {"KEEP": 4, "NEW": 1},
        include_stock=True,
        sync_date="2026-06-09 12:00 UTC",
    )
    assert written2 == 0
    assert removed2 == 0


def test_sync_catalog_files_writes_price_from_price_list(
    test_settings: SettingsForTests,
) -> None:
    slug = "catalog-price-bot"
    items = [{"item_code": "8.2.7", "item_name": "Keyboard", "standard_rate": 0, "stock_uom": "Nos"}]
    written, _, _ = sync_catalog_files(
        test_settings,
        slug,
        items,
        {},
        include_stock=False,
        price_by_code={
            "8.2.7": {
                "rate": 34176.0,
                "currency": "MUR",
                "price_list": "Standard Selling",
            }
        },
        sync_date="2026-06-09 12:00 UTC",
    )
    assert written == 1
    content = (tenant_catalog_dir(test_settings, slug) / "8.2.7.md").read_text(encoding="utf-8")
    assert "Price: 34176 MUR (Standard Selling)" in content


def test_catalog_sync_due_respects_interval() -> None:
    now = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    cfg = {
        "sync_catalog_to_rag": True,
        "catalog_sync_interval_minutes": 60,
        "catalog_last_sync_at": (now - timedelta(minutes=30)).isoformat(),
    }
    assert catalog_sync_due(cfg, now=now) is False
    cfg["catalog_last_sync_at"] = (now - timedelta(minutes=61)).isoformat()
    assert catalog_sync_due(cfg, now=now) is True
    assert catalog_sync_enabled(cfg) is True


def test_sync_erpnext_catalog_for_tenant_batches_rag(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    tenant, _ = test_tenant
    client = MagicMock()
    client.list_catalog_items.return_value = [
        {
            "item_code": "ITEM-1",
            "item_name": "One",
            "standard_rate": 1,
            "stock_uom": "Nos",
        }
    ]
    client.fetch_stock_totals.return_value = {"ITEM-1": 3}
    client.fetch_price_list_rates.return_value = {}

    ingest_calls: list[int] = []

    def fake_batched(self, paths, *, batch_size=100, pause_seconds=0.0):
        ingest_calls.append(len(paths))
        return [f"ingested:{len(paths)}"]

    with patch.object(IngestSyncService, "ingest_paths_batched", fake_batched), patch(
        "chatbot.application.erpnext_catalog_sync_service.GeminiEmbedder"
    ), patch("chatbot.application.erpnext_catalog_sync_service.LanceVectorStore"):
        engine = create_db_engine(test_settings, for_tests=True)
        factory = session_factory(engine)
        with factory() as session:
            result = sync_erpnext_catalog_for_tenant(
                session,
                settings=test_settings,
                tenant_id=tenant.id,
                tenant_slug=tenant.slug,
                config={"sync_catalog_to_rag": True},
                client=client,
                batch_size=100,
            )
        engine.dispose()

    assert result.ok is True
    assert result.item_count == 1
    assert result.files_written == 1
    assert ingest_calls == [1]
    path = tenant_catalog_dir(test_settings, tenant.slug) / "ITEM-1.md"
    assert path.is_file()
    client.fetch_price_list_rates.assert_called_once_with("Standard Selling")


def test_sync_skips_price_fetch_when_price_list_blank(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    tenant, _ = test_tenant
    client = MagicMock()
    client.list_catalog_items.return_value = [
        {"item_code": "ITEM-1", "item_name": "One", "standard_rate": 5, "stock_uom": "Nos"}
    ]
    client.fetch_stock_totals.return_value = {}

    with patch.object(IngestSyncService, "ingest_paths_batched", return_value=[]), patch(
        "chatbot.application.erpnext_catalog_sync_service.GeminiEmbedder"
    ), patch("chatbot.application.erpnext_catalog_sync_service.LanceVectorStore"):
        engine = create_db_engine(test_settings, for_tests=True)
        factory = session_factory(engine)
        with factory() as session:
            sync_erpnext_catalog_for_tenant(
                session,
                settings=test_settings,
                tenant_id=tenant.id,
                tenant_slug=tenant.slug,
                config={"sync_catalog_to_rag": True, "catalog_price_list": ""},
                client=client,
            )
        engine.dispose()

    client.fetch_price_list_rates.assert_not_called()
