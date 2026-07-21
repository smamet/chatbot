from __future__ import annotations

from unittest.mock import MagicMock

from chatbot.application.quote_sync_state import (
    normalize_erpnext_modified,
    quote_pdf_is_stale,
    quote_pdf_stale_context,
    quotation_erp_modified,
)


def test_normalize_erpnext_modified_strips_microseconds() -> None:
    assert normalize_erpnext_modified("2026-06-15 14:17:39.476598") == "2026-06-15 14:17:39"


def test_quote_pdf_is_stale_when_modified_differs() -> None:
    assert quote_pdf_is_stale(
        "2026-06-15 14:17:39.476598",
        "2026-06-15 14:18:15.466254",
    )
    assert not quote_pdf_is_stale(
        "2026-06-15 14:17:39.476598",
        "2026-06-15 14:17:39.999999",
    )
    assert not quote_pdf_is_stale(None, "2026-06-15 14:17:39")
    assert not quote_pdf_is_stale("2026-06-15 14:17:39", None)


def test_quotation_erp_modified_reads_doc() -> None:
    client = MagicMock()
    client.get_quotation.return_value = {"modified": "2026-06-15 14:17:39.476598"}
    assert quotation_erp_modified(client, "QTN-0001") == "2026-06-15 14:17:39.476598"


def test_quote_pdf_stale_context() -> None:
    client = MagicMock()
    client.get_quotation.return_value = {"modified": "2026-06-15 14:18:15"}
    ctx = quote_pdf_stale_context(
        client=client,
        tenant_slug="bot",
        quote_name="QTN-0001",
        stored_modified="2026-06-15 14:17:39",
        erpnext_url="https://erp.example.com/app/quotation/QTN-0001",
    )
    assert ctx["stale"] is True
    assert ctx["download_url"].endswith("?inline=1")
    assert ctx["erpnext_url"].startswith("https://erp")
