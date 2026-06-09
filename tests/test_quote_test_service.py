from __future__ import annotations

from unittest.mock import MagicMock, patch

from chatbot.application.progress_log import ProgressLog
from chatbot.application.quote_test_service import create_erpnext_quotation_for_test
from chatbot.config.settings import get_settings


def test_create_quotation_disabled() -> None:
    client = MagicMock()
    settings = get_settings()
    out = create_erpnext_quotation_for_test(
        client,
        {"allow_create_quotation": False},
        settings=settings,
        tenant_slug="bot",
        test_email="a@example.com",
        test_phone="",
        item_code="SKU-1",
        qty=1,
    )
    assert out["ok"] is False
    assert out["error"] == "creation_disabled"
    client.create_quotation.assert_not_called()


def test_create_quotation_happy_path() -> None:
    client = MagicMock()
    client.find_customer.return_value = "Alice Corp"
    client.resolve_item.return_value = {"item_code": "SKU-1", "standard_rate": 12.5}
    client.create_quotation.return_value = {"name": "QTN-0001"}
    client.download_quotation_pdf.return_value = b"%PDF"
    settings = get_settings()

    with patch("chatbot.application.quote_test_service.store_quote_pdf") as store_pdf:
        store_pdf.return_value = settings.data_root / "quotes" / "bot" / "QTN-0001.pdf"
        out = create_erpnext_quotation_for_test(
            client,
            {"allow_create_quotation": True},
            settings=settings,
            tenant_slug="bot",
            test_email="alice@example.com",
            test_phone="",
            item_code="SKU-1",
            qty=2,
            notes="Test",
        )

    assert out["ok"] is True
    assert out["quote_name"] == "QTN-0001"
    assert out["customer"] == "Alice Corp"
    assert out["pdf_url"] == "/dashboard/bots/bot/integrations/erpnext/quotation-pdf/QTN-0001"
    assert out["pdf_filename"] == "QTN-0001.pdf"
    client.create_quotation.assert_called_once_with(
        "Alice Corp",
        [{"item_code": "SKU-1", "qty": 2, "rate": 12.5}],
        notes="Test",
    )


def test_create_quotation_emits_progress_logs() -> None:
    client = MagicMock()
    client.find_customer.return_value = "Alice Corp"
    client.resolve_item.return_value = {"item_code": "SKU-1", "standard_rate": 12.5}
    client.create_quotation.return_value = {"name": "QTN-0001"}
    client.download_quotation_pdf.return_value = b"%PDF"
    settings = get_settings()
    progress = ProgressLog()

    with patch("chatbot.application.quote_test_service.store_quote_pdf"):
        create_erpnext_quotation_for_test(
            client,
            {"allow_create_quotation": True},
            settings=settings,
            tenant_slug="bot",
            test_email="alice@example.com",
            test_phone="",
            item_code="SKU-1",
            qty=1,
            on_log=progress,
        )

    assert len(progress.messages) >= 3
    assert any("Customer found" in msg for msg in progress.messages)
    assert any("Quotation created" in msg for msg in progress.messages)
    client.download_quotation_pdf.assert_called_once()
    assert client.download_quotation_pdf.call_args.kwargs.get("on_log") is progress
