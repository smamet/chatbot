from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chatbot.application.catalog_inspector_service import (
    RagCatalogRow,
    _description_for_table,
    _format_converted_hint,
    _plain_text,
    _price_amount,
    _price_source_label,
    build_inspector_page,
    filter_by_rag_price,
    filter_rows,
    filter_by_mismatch,
    load_rag_rows,
    merge_inspector_rows,
    normalize_mismatch_filter,
    normalize_price_filter,
    parse_catalog_markdown,
    read_invoice_cache,
    write_invoice_cache,
)
from chatbot.application.erpnext_catalog_sync_service import render_item_markdown


SAMPLE_MD = render_item_markdown(
    {
        "item_code": "W-1",
        "item_name": "Widget",
        "description": "A useful widget for testing.",
        "standard_rate": 0,
        "stock_uom": "Nos",
        "item_group": "Products",
    },
    stock_qty=3,
    sync_date="2026-06-23 10:00 UTC",
    include_stock=True,
    price_entry={"rate": 1500.0, "currency": "MUR", "price_list": "Standard Selling"},
)


def test_parse_catalog_markdown_item_price() -> None:
    row = parse_catalog_markdown(SAMPLE_MD)
    assert row is not None
    assert row.item_code == "W-1"
    assert row.name == "Widget"
    assert row.description == "A useful widget for testing."
    assert row.price_rate == 1500.0
    assert row.price_source == "Standard Selling"
    assert "1,500 MUR" in row.price_display


def test_parse_catalog_markdown_standard_rate_and_not_available() -> None:
    std_md = render_item_markdown(
        {"item_code": "S-1", "item_name": "Std", "standard_rate": 42, "stock_uom": "Nos"},
        stock_qty=None,
        sync_date="2026-06-23 10:00 UTC",
        include_stock=False,
    )
    std_row = parse_catalog_markdown(std_md)
    assert std_row is not None
    assert std_row.price_rate == 42.0
    assert std_row.price_source == "standard_rate"

    missing_md = render_item_markdown(
        {"item_code": "X-1", "item_name": "Missing", "standard_rate": 0, "stock_uom": "Nos"},
        stock_qty=None,
        sync_date="2026-06-23 10:00 UTC",
        include_stock=False,
    )
    missing_row = parse_catalog_markdown(missing_md)
    assert missing_row is not None
    assert missing_row.price_display == "not available"
    assert missing_row.price_rate is None


def test_load_rag_rows_from_directory(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "W-1.md").write_text(SAMPLE_MD, encoding="utf-8")
    rows = load_rag_rows(catalog)
    assert "W-1" in rows
    assert rows["W-1"].name == "Widget"


def test_merge_inspector_rows_detects_mismatch() -> None:
    rag_rows = {
        "W-1": RagCatalogRow(
            item_code="W-1",
            name="Widget",
            description="Desc",
            price_display="not available",
            price_source=None,
            price_rate=None,
            price_currency=None,
        )
    }
    item_prices = {
        "W-1": {"rate": 1500.0, "currency": "MUR", "price_list": "Standard Selling"},
    }
    rows = merge_inspector_rows(
        rag_rows,
        item_prices=item_prices,
        standard_rates={},
        invoice_cache=None,
        config={},
    )
    assert len(rows) == 1
    assert rows[0].mismatch is True
    assert rows[0].item_price_display.startswith("1500")


def test_filter_rows_matches_description() -> None:
    from chatbot.application.catalog_inspector_service import InspectorRow

    rows = [
        InspectorRow(
            item_code="A",
            name="Alpha",
            md_filename="A.md",
            description="special keyword here",
            description_truncated="special…",
            description_expandable=True,
            rag_price_display="10",
            rag_source=None,
            item_price_display="—",
            standard_rate_display="—",
            invoice_price_display="—",
            rag_price_converted_display=None,
            item_price_converted_display=None,
            invoice_price_converted_display=None,
            mismatch=False,
            expected_source=None,
        ),
        InspectorRow(
            item_code="B",
            name="Beta",
            md_filename="B.md",
            description="other",
            description_truncated="other",
            description_expandable=False,
            rag_price_display="20",
            rag_source=None,
            item_price_display="—",
            standard_rate_display="—",
            invoice_price_display="—",
            rag_price_converted_display=None,
            item_price_converted_display=None,
            invoice_price_converted_display=None,
            mismatch=False,
            expected_source=None,
        ),
    ]
    filtered = filter_rows(rows, "keyword")
    assert len(filtered) == 1
    assert filtered[0].item_code == "A"


def _inspector_row(
    item_code: str,
    *,
    rag_price_display: str = "not available",
) -> "InspectorRow":
    from chatbot.application.catalog_inspector_service import InspectorRow

    return InspectorRow(
        item_code=item_code,
        name=item_code,
        md_filename=f"{item_code}.md",
        description="",
        description_truncated="",
        description_expandable=False,
        rag_price_display=rag_price_display,
        rag_source=None,
        item_price_display="—",
        standard_rate_display="—",
        invoice_price_display="—",
        rag_price_converted_display=None,
        item_price_converted_display=None,
        invoice_price_converted_display=None,
        mismatch=False,
        expected_source=None,
    )


def test_filter_by_rag_price() -> None:
    rows = [
        _inspector_row("A", rag_price_display="10"),
        _inspector_row("B", rag_price_display="not available"),
        _inspector_row("C", rag_price_display="20 MUR"),
    ]
    assert [row.item_code for row in filter_by_rag_price(rows, "with")] == ["A", "C"]
    assert [row.item_code for row in filter_by_rag_price(rows, "without")] == ["B"]
    assert [row.item_code for row in filter_by_rag_price(rows, "all")] == ["A", "B", "C"]
    assert normalize_price_filter("invalid") == "all"


def _inspector_row_mismatch(item_code: str, *, mismatch: bool) -> "InspectorRow":
    from chatbot.application.catalog_inspector_service import InspectorRow

    return InspectorRow(
        item_code=item_code,
        name=item_code,
        md_filename=f"{item_code}.md",
        description="",
        description_truncated="",
        description_expandable=False,
        rag_price_display="10",
        rag_source=None,
        item_price_display="—",
        standard_rate_display="—",
        invoice_price_display="—",
        rag_price_converted_display=None,
        item_price_converted_display=None,
        invoice_price_converted_display=None,
        mismatch=mismatch,
        expected_source=None,
    )


def test_filter_by_mismatch() -> None:
    rows = [
        _inspector_row_mismatch("A", mismatch=True),
        _inspector_row_mismatch("B", mismatch=False),
        _inspector_row_mismatch("C", mismatch=True),
    ]
    assert [row.item_code for row in filter_by_mismatch(rows, "mismatch")] == ["A", "C"]
    assert [row.item_code for row in filter_by_mismatch(rows, "aligned")] == ["B"]
    assert [row.item_code for row in filter_by_mismatch(rows, "all")] == ["A", "B", "C"]
    assert normalize_mismatch_filter("invalid") == "all"


def test_price_amount_strips_source_suffix() -> None:
    assert _price_amount("1200 MUR (Standard Selling)") == "1200 MUR"
    assert _price_source_label("1200 MUR (Standard Selling)") == "Standard Selling"
    assert _price_amount("not available") == "not available"


def test_description_for_table_strips_html_and_duplicate_title() -> None:
    html = "<div>1 E2ads v5 (2 vCPUs, 16 GB RAM) x 730 Hours</div>"
    preview = _description_for_table("1 E2ads v5", html)
    assert preview.startswith("1 E2ads v5 (2 vCPUs")
    assert "<div>" not in preview
    assert _description_for_table("AXIS Q6358-LE", "AXIS Q6358-LE") == "AXIS Q6358-LE"
    assert _description_for_table("Widget", "Widget details") == "Widget details"
    dynamic = _description_for_table(
        "1 Dynamic IP Addresses",
        "<div>1 Dynamic IP Addresses, 1 Static IP Addresses, 0 Remaps</div>",
    )
    assert dynamic.startswith("1 Dynamic IP Addresses, 1 Static")


def test_format_numeric_amount_uses_thousands_separator() -> None:
    from chatbot.application.catalog_inspector_service import _format_numeric_amount

    assert _format_numeric_amount(2492.01) == "2,492.01"
    assert _format_numeric_amount(119270.47) == "119,270.47"
    assert _format_numeric_amount(1500.0) == "1,500"


def test_format_converted_hint_skips_same_currency() -> None:
    from unittest.mock import MagicMock

    fx = MagicMock()
    assert _format_converted_hint(100.0, "MUR", fx=fx, compare_base="MUR") is None
    fx.convert.assert_not_called()


def test_format_converted_hint_shows_target_amount() -> None:
    from unittest.mock import MagicMock

    fx = MagicMock()
    fx.convert.return_value = 115000.0
    hint = _format_converted_hint(2492.01, "USD", fx=fx, compare_base="MUR")
    assert hint == "≈ 115,000 MUR"
    fx.convert.assert_called_once_with(2492.01, "USD", "MUR")


def test_merge_inspector_shows_fx_conversion_for_usd() -> None:
    from unittest.mock import MagicMock

    from chatbot.application.catalog_inspector_service import InvoicePriceCache

    fx = MagicMock()
    fx.convert.return_value = 115000.0
    rag_rows = {
        "W-1": RagCatalogRow(
            item_code="W-1",
            name="Widget",
            description="",
            price_display="2492.01 USD (last invoice)",
            price_source="last invoice",
            price_rate=2492.01,
            price_currency="USD",
        )
    }
    invoice_cache = InvoicePriceCache(
        cached_at="2026-06-24T12:00:00+00:00",
        rates={"W-1": {"rate": 2492.01, "currency": "USD"}},
    )
    rows = merge_inspector_rows(
        rag_rows,
        item_prices={},
        standard_rates={},
        invoice_cache=invoice_cache,
        config={"catalog_price_compare_currency": "MUR"},
        fx=fx,
    )
    assert rows[0].rag_price_converted_display == "≈ 115,000 MUR"
    assert rows[0].invoice_price_converted_display == "≈ 115,000 MUR"
    assert rows[0].item_price_converted_display is None


def test_merge_inspector_sets_md_filename() -> None:
    rag_rows = {
        "W-1": RagCatalogRow(
            item_code="W-1",
            name="Widget",
            description="Widget details",
            price_display="10 MUR (Standard Selling)",
            price_source="Standard Selling",
            price_rate=10.0,
            price_currency="MUR",
        )
    }
    rows = merge_inspector_rows(
        rag_rows,
        item_prices={},
        standard_rates={},
        invoice_cache=None,
        config={},
    )
    assert rows[0].md_filename == "W-1.md"
    assert rows[0].description_truncated == "Widget details"


def test_invoice_cache_roundtrip(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    cache = write_invoice_cache(catalog, {"A": {"rate": 99.0, "currency": "MUR"}})
    loaded = read_invoice_cache(catalog)
    assert loaded is not None
    assert loaded.cached_at == cache.cached_at
    assert loaded.rates["A"]["rate"] == 99.0


def test_build_inspector_page_with_mocked_erp(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "W-1.md").write_text(SAMPLE_MD, encoding="utf-8")
    client = MagicMock()
    client.fetch_price_list_rates.return_value = {
        "W-1": {"rate": 1500.0, "currency": "MUR", "price_list": "Standard Selling"},
    }
    client.list_catalog_items.return_value = [
        {"item_code": "W-1", "standard_rate": 0},
    ]
    page = build_inspector_page(
        catalog,
        client=client,
        config={"catalog_price_list": "Standard Selling"},
        query="widget",
        page=1,
    )
    assert page.total == 1
    assert page.rows[0].item_code == "W-1"
    assert page.rows[0].mismatch is False


def test_expected_price_entry_highest_price_picks_invoice() -> None:
    from unittest.mock import MagicMock

    from chatbot.application.catalog_inspector_service import (
        InvoicePriceCache,
        _expected_price_entry,
    )

    item_prices = {
        "A": {"rate": 100.0, "currency": "MUR", "price_list": "Standard Selling"},
    }
    invoice_cache = InvoicePriceCache(
        cached_at="2026-06-24T12:00:00+00:00",
        rates={"A": {"rate": 200.0, "currency": "MUR"}},
    )
    fx = MagicMock()
    fx.convert.return_value = None
    display, source, rate = _expected_price_entry(
        "A",
        item_prices=item_prices,
        standard_rates={},
        invoice_cache=invoice_cache,
        config={"catalog_use_highest_price": True},
        fx=fx,
    )
    assert rate == 200.0
    assert source == "last invoice"
