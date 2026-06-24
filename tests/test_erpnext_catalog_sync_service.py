from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_paths import safe_catalog_filename, tenant_catalog_dir
from chatbot.application.erpnext_catalog_sync_service import (
    build_catalog_price_map,
    catalog_invoice_price_fallback,
    catalog_price_list,
    catalog_use_highest_price,
    pick_catalog_price_entry,
    catalog_rag_index_plan,
    catalog_sync_due,
    catalog_sync_enabled,
    render_item_markdown,
    reconcile_catalog_rag,
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
    assert catalog_price_list({"catalog_price_list": ""}) == "Standard Selling"
    assert catalog_price_list({"catalog_price_list": "  "}) == "Standard Selling"


def test_catalog_invoice_price_fallback_defaults_false() -> None:
    assert catalog_invoice_price_fallback({}) is False
    assert catalog_invoice_price_fallback({"catalog_invoice_price_fallback": True}) is True
    assert catalog_invoice_price_fallback({"catalog_use_highest_price": True}) is False
    assert catalog_invoice_price_fallback({"catalog_invoice_price_fallback": "false"}) is False


def test_catalog_use_highest_price_defaults_false() -> None:
    assert catalog_use_highest_price({}) is False
    assert catalog_use_highest_price({"catalog_use_highest_price": True}) is True


def test_pick_catalog_price_entry_same_currency_picks_higher() -> None:
    item = {"rate": 175.0, "currency": "MUR", "price_list": "Standard Selling"}
    invoice = {"rate": 2492.0, "currency": "USD", "price_list": "last invoice"}
    fx = MagicMock()
    fx.convert.side_effect = lambda amount, src, dst: (
        amount if src == dst else (amount * 50.0 if src == "USD" and dst == "MUR" else None)
    )
    picked = pick_catalog_price_entry(
        item_price_entry=item,
        invoice_entry=invoice,
        standard_rate=None,
        fx=fx,
        compare_base="MUR",
    )
    assert picked is invoice


def test_pick_catalog_price_entry_single_candidate() -> None:
    item = {"rate": 10.0, "currency": "MUR", "price_list": "Standard Selling"}
    picked = pick_catalog_price_entry(
        item_price_entry=item,
        invoice_entry=None,
        standard_rate=None,
        fx=None,
        compare_base="MUR",
    )
    assert picked is item


def test_build_catalog_price_map_merges_invoice_fallback() -> None:
    client = MagicMock()
    client.fetch_price_list_rates.return_value = {
        "A": {"rate": 10.0, "price_list": "Standard Selling"},
    }
    client.fetch_latest_invoice_rates.return_value = {
        "B": {"rate": 99.0, "price_list": "last invoice"},
    }
    items = [
        {"item_code": "A", "standard_rate": 0},
        {"item_code": "B", "standard_rate": 0},
        {"item_code": "C", "standard_rate": 5},
    ]
    prices = build_catalog_price_map(
        items,
        erp_client=client,
        config={"catalog_invoice_price_fallback": True},
    )
    assert prices["A"]["rate"] == 10.0
    assert prices["B"]["rate"] == 99.0
    assert "C" not in prices
    client.fetch_latest_invoice_rates.assert_called_once_with(item_codes={"B"})


def test_build_catalog_price_map_skips_invoice_when_disabled() -> None:
    client = MagicMock()
    client.fetch_price_list_rates.return_value = {}
    items = [{"item_code": "B", "standard_rate": 0}]
    prices = build_catalog_price_map(
        items,
        erp_client=client,
        config={"catalog_invoice_price_fallback": False},
    )
    assert prices == {}
    client.fetch_latest_invoice_rates.assert_not_called()


def test_build_catalog_price_map_highest_price_mode() -> None:
    client = MagicMock()
    client.fetch_price_list_rates.return_value = {
        "A": {"rate": 100.0, "currency": "MUR", "price_list": "Standard Selling"},
    }
    client.fetch_latest_invoice_rates.return_value = {
        "A": {"rate": 200.0, "currency": "MUR", "price_list": "last invoice"},
    }
    items = [{"item_code": "A", "standard_rate": 0}]
    fx = MagicMock()
    fx.convert.return_value = None
    prices = build_catalog_price_map(
        items,
        erp_client=client,
        config={"catalog_use_highest_price": True},
        fx=fx,
    )
    assert prices["A"]["rate"] == 200.0
    client.fetch_latest_invoice_rates.assert_called_once_with()


def test_render_item_markdown_uses_last_invoice_price() -> None:
    md = render_item_markdown(
        {
            "item_code": "X-1",
            "item_name": "Widget",
            "standard_rate": 0,
            "stock_uom": "Nos",
        },
        stock_qty=None,
        sync_date="2026-06-09 12:00 UTC",
        include_stock=False,
        price_entry={
            "rate": 2500.0,
            "currency": "MUR",
            "price_list": "last invoice",
        },
    )
    assert "Price: 2500 MUR (last invoice)" in md


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

    result1 = sync_catalog_files(
        test_settings,
        slug,
        items,
        {"KEEP": 4, "NEW": 1},
        include_stock=True,
        sync_date="2026-06-09 12:00 UTC",
    )
    assert result1.written == 2
    assert result1.removed == 1
    assert len(result1.changed_paths) == 2
    assert not stale.exists()
    assert (root / "KEEP.md").is_file()
    assert (root / "NEW.md").is_file()

    result2 = sync_catalog_files(
        test_settings,
        slug,
        items,
        {"KEEP": 4, "NEW": 1},
        include_stock=True,
        sync_date="2026-06-09 12:00 UTC",
    )
    assert result2.written == 0
    assert result2.removed == 0
    assert result2.changed_paths == []


def test_sync_catalog_files_skips_write_when_only_sync_date_changes(
    test_settings: SettingsForTests,
) -> None:
    slug = "catalog-sync-date-bot"
    items = [
        {
            "item_code": "KEEP",
            "item_name": "Keep me",
            "standard_rate": 1,
            "stock_uom": "Nos",
        },
    ]
    root = tenant_catalog_dir(test_settings, slug)

    result1 = sync_catalog_files(
        test_settings,
        slug,
        items,
        {"KEEP": 4},
        include_stock=True,
        sync_date="2026-06-14 10:00 UTC",
    )
    assert result1.written == 1
    content = (root / "KEEP.md").read_text(encoding="utf-8")
    assert "Stock/price as of: 2026-06-14 10:00 UTC" in content

    result2 = sync_catalog_files(
        test_settings,
        slug,
        items,
        {"KEEP": 4},
        include_stock=True,
        sync_date="2026-06-14 11:00 UTC",
    )
    assert result2.written == 0
    assert result2.removed == 0
    assert result2.changed_paths == []
    assert "Stock/price as of: 2026-06-14 10:00 UTC" in (root / "KEEP.md").read_text(
        encoding="utf-8"
    )


def test_sync_catalog_files_writes_price_from_price_list(
    test_settings: SettingsForTests,
) -> None:
    slug = "catalog-price-bot"
    items = [{"item_code": "8.2.7", "item_name": "Keyboard", "standard_rate": 0, "stock_uom": "Nos"}]
    result = sync_catalog_files(
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
    assert result.written == 1
    content = (tenant_catalog_dir(test_settings, slug) / "8.2.7.md").read_text(encoding="utf-8")
    assert "Price: 34176 MUR (Standard Selling)" in content


def test_sync_catalog_files_duplicate_item_codes_are_stable(
    test_settings: SettingsForTests,
) -> None:
    slug = "catalog-dup-bot"
    items = [
        {
            "item_code": "DS-7616NXI-K2/16P",
            "item_name": "Slash code",
            "standard_rate": 1,
            "stock_uom": "Nos",
        },
        {
            "item_code": "DS-7616NXI-K2-16P",
            "item_name": "Dash code",
            "standard_rate": 1,
            "stock_uom": "Nos",
        },
    ]
    root = tenant_catalog_dir(test_settings, slug)
    path = root / "DS-7616NXI-K2-16P.md"

    result1 = sync_catalog_files(
        test_settings,
        slug,
        items,
        {},
        include_stock=False,
        sync_date="2026-06-14 10:00 UTC",
    )
    assert result1.written == 1
    first = path.read_text(encoding="utf-8")

    result2 = sync_catalog_files(
        test_settings,
        slug,
        items,
        {},
        include_stock=False,
        sync_date="2026-06-14 11:00 UTC",
    )
    assert result2.written == 0
    assert path.read_text(encoding="utf-8") == first
    assert any("shared by 2 item codes" in line for line in result1.logs)


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

    def fake_batched(
        self,
        paths,
        *,
        batch_size=100,
        pause_seconds=0.0,
        commit_each_batch=False,
        on_file_done=None,
    ):
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


def test_sync_erpnext_catalog_skips_rag_when_no_file_changes(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    tenant, _ = test_tenant
    slug = tenant.slug
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

    with patch.object(IngestSyncService, "ingest_paths_batched") as mock_ingest, patch.object(
        IngestSyncService, "maybe_optimize", return_value=[]
    ), patch(
        "chatbot.application.erpnext_catalog_sync_service.GeminiEmbedder"
    ), patch("chatbot.application.erpnext_catalog_sync_service.LanceVectorStore"):
        engine = create_db_engine(test_settings, for_tests=True)
        factory = session_factory(engine)
        with factory() as session:
            sync_erpnext_catalog_for_tenant(
                session,
                settings=test_settings,
                tenant_id=tenant.id,
                tenant_slug=slug,
                config={"sync_catalog_to_rag": True},
                client=client,
            )
            mock_ingest.reset_mock()
            result = sync_erpnext_catalog_for_tenant(
                session,
                settings=test_settings,
                tenant_id=tenant.id,
                tenant_slug=slug,
                config={"sync_catalog_to_rag": True},
                client=client,
            )
        engine.dispose()

    assert result.ok is True
    assert result.files_written == 0
    assert result.rag_files_indexed == 0
    assert "RAG ingest skipped" in result.message
    mock_ingest.assert_not_called()


def test_sync_uses_standard_selling_when_price_list_blank(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    tenant, _ = test_tenant
    client = MagicMock()
    client.list_catalog_items.return_value = [
        {"item_code": "ITEM-1", "item_name": "One", "standard_rate": 5, "stock_uom": "Nos"}
    ]
    client.fetch_stock_totals.return_value = {}
    client.fetch_price_list_rates.return_value = {}
    client.fetch_latest_invoice_rates.return_value = {}

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

    client.fetch_price_list_rates.assert_called_once_with("Standard Selling")


def test_reconcile_catalog_rag_skips_ingest_when_no_paths(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    tenant, _ = test_tenant
    catalog_dir = tenant_catalog_dir(test_settings, tenant.slug)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "ITEM-1.md").write_text("# One\n", encoding="utf-8")

    with patch.object(
        IngestSyncService,
        "ingest_paths_batched",
        return_value=["should-not-run"],
    ) as mock_ingest, patch(
        "chatbot.application.erpnext_catalog_sync_service.GeminiEmbedder"
    ), patch("chatbot.application.erpnext_catalog_sync_service.LanceVectorStore"):
        engine = create_db_engine(test_settings, for_tests=True)
        factory = session_factory(engine)
        with factory() as session:
            logs = reconcile_catalog_rag(
                session,
                settings=test_settings,
                tenant_id=tenant.id,
                slug=tenant.slug,
                paths_to_reindex=[],
            )
        engine.dispose()

    mock_ingest.assert_not_called()
    assert any("skipped RAG ingest" in line for line in logs)


def test_reconcile_catalog_rag_calls_optimize(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    tenant, _ = test_tenant
    catalog_dir = tenant_catalog_dir(test_settings, tenant.slug)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "ITEM-1.md").write_text("# One\n", encoding="utf-8")

    with patch.object(
        IngestSyncService,
        "ingest_paths_batched",
        return_value=["unchanged: item"],
    ), patch.object(
        IngestSyncService,
        "maybe_optimize",
        return_value=["optimized LanceDB table (stats)"],
    ) as mock_optimize, patch(
        "chatbot.application.erpnext_catalog_sync_service.GeminiEmbedder"
    ), patch("chatbot.application.erpnext_catalog_sync_service.LanceVectorStore"):
        engine = create_db_engine(test_settings, for_tests=True)
        factory = session_factory(engine)
        with factory() as session:
            logs = reconcile_catalog_rag(
                session,
                settings=test_settings,
                tenant_id=tenant.id,
                slug=tenant.slug,
                paths_to_reindex=[catalog_dir / "ITEM-1.md"],
            )
        engine.dispose()

    mock_optimize.assert_called_once()
    assert "optimized LanceDB table" in logs[-1]


def test_catalog_rag_index_plan_splits_missing_and_indexed(
    test_settings: SettingsForTests,
    test_tenant,
) -> None:
    from chatbot.adapters.persistence.orm import IngestedFileRow
    from chatbot.application.ingest_service import file_content_hash

    tenant, _ = test_tenant
    catalog_dir = tenant_catalog_dir(test_settings, tenant.slug)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    indexed_path = catalog_dir / "indexed.md"
    missing_path = catalog_dir / "missing.md"
    indexed_path.write_text("# indexed\n", encoding="utf-8")
    missing_path.write_text("# missing\n", encoding="utf-8")

    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    with factory() as session:
        session.add(
            IngestedFileRow(
                tenant_id=tenant.id,
                path=str(indexed_path.resolve()),
                content_hash=file_content_hash(indexed_path),
            )
        )
        session.commit()
        plan = catalog_rag_index_plan(session, tenant.id, catalog_dir)
    engine.dispose()

    assert missing_path in plan.needs_embed
    assert indexed_path in plan.already_indexed


def test_apply_catalog_rag_transition_purges_on_disable(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    catalog_dir = tenant_catalog_dir(test_settings, tenant.slug)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    md_path = catalog_dir / "item.md"
    md_path.write_text("# item", encoding="utf-8")

    with factory() as session:
        from chatbot.adapters.persistence.orm import IngestedFileRow

        session.add(
            IngestedFileRow(
                tenant_id=tenant.id,
                path=str(md_path),
                content_hash="abc",
            )
        )
        session.commit()

        deleted: list[str] = []

        class FakeStore:
            def delete_by_source_path(self, path: str) -> None:
                deleted.append(path)

        with patch(
            "chatbot.application.erpnext_catalog_sync_service.LanceVectorStore",
            return_value=FakeStore(),
        ), patch(
            "chatbot.application.erpnext_catalog_sync_service.GeminiEmbedder",
        ):
            from chatbot.application.erpnext_catalog_sync_service import apply_catalog_rag_transition

            apply_catalog_rag_transition(
                session,
                test_settings,
                tenant_id=tenant.id,
                tenant_slug=tenant.slug,
                integration_id=1,
                config={"sync_catalog_to_rag": False},
                prev_enabled=True,
                now_enabled=False,
                run_sync_background=lambda *a, **k: None,
            )
            session.commit()

    assert md_path.is_file()
    assert str(md_path) in deleted


def test_apply_catalog_rag_transition_starts_import_on_enable(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    started: list[tuple] = []

    def fake_sync(settings, **kwargs):
        started.append((settings, kwargs))

    with factory() as session:
        from chatbot.application.erpnext_catalog_sync_service import apply_catalog_rag_transition

        apply_catalog_rag_transition(
            session,
            test_settings,
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            integration_id=42,
            config={"sync_catalog_to_rag": True},
            prev_enabled=False,
            now_enabled=True,
            run_sync_background=fake_sync,
        )
    assert len(started) == 1
    assert started[0][1]["integration_id"] == 42
