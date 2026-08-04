from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from evenor.application.customer_provisioning_service import CustomerProvisioningError
from evenor.application.quote_fulfillment_service import QuoteFulfillmentError, QuoteFulfillmentService
from evenor.config.settings import get_settings
from evenor.domain.models.fulfillment import FulfillmentKind
from evenor.domain.models.pending_reply import PendingReply, PendingReplyStatus


def _quote_reply(*, session_id: str = "email:alice@example.com") -> PendingReply:
    now = datetime.now(UTC)
    return PendingReply(
        id=1,
        tenant_id=1,
        connector_id=1,
        session_id=session_id,
        channel="email",
        recipient_id="alice@example.com",
        draft_text="Here is your quote",
        status=PendingReplyStatus.PENDING,
        created_at=now,
        updated_at=now,
        fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
        quote_proposal_json=json.dumps(
            {"type": "quote.create", "lines": [{"product": "Widget", "qty": 1}]}
        ),
        quote_resolved_json=json.dumps(
            [
                {
                    "requested_label": "Widget",
                    "qty": 1,
                    "item_code": "SKU-1",
                    "status": "resolved",
                    "rate": 10.0,
                }
            ]
        ),
    )


def test_fulfill_blocks_when_quotation_creation_disabled() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    reply = _quote_reply()
    integration_config = {"allow_create_quotation": False, "allow_create_customer": True}
    client = MagicMock()

    with patch(
        "evenor.application.quote_fulfillment_service.erpnext_integration_for_tenant",
        return_value=(client, integration_config),
    ):
        with pytest.raises(QuoteFulfillmentError, match="Quotation creation is disabled"):
            svc.fulfill_and_approve(reply, config={})


def test_fulfill_creates_customer_submits_and_redownloads_pdf() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    reply = _quote_reply()
    integration_config = {"allow_create_quotation": True, "allow_create_customer": True}
    client = MagicMock()
    client.find_customer.return_value = None
    client.create_quotation.return_value = {"name": "QTN-0001"}
    client.submit_quotation.return_value = None
    client.download_quotation_pdf.return_value = b"%PDF"

    with patch(
        "evenor.application.quote_fulfillment_service.erpnext_integration_for_tenant",
        return_value=(client, integration_config),
    ), patch(
        "evenor.application.quote_fulfillment_service.ensure_erpnext_customer",
        return_value="New Customer",
    ) as ensure_mock, patch(
        "evenor.application.quote_fulfillment_service.quotation_erp_modified",
        return_value="2026-06-15 14:17:39",
    ), patch(
        "evenor.application.quote_fulfillment_service.approve_pending_reply",
        return_value=reply,
    ), patch(
        "evenor.application.quote_fulfillment_service.SqlAlchemyPendingReplyRepository"
    ) as repo_cls, patch(
        "evenor.application.quote_fulfillment_service.delete_attachment_files",
    ) as delete_files, patch(
        "evenor.application.quote_fulfillment_service.store_quote_pdf",
        return_value=settings.data_root / "quotes" / "bot" / "QTN-0001.pdf",
    ):
        repo = repo_cls.return_value
        svc.fulfill_and_approve(reply, config={})
        ensure_mock.assert_called_once()
        client.create_quotation.assert_called_once_with(
            "New Customer",
            [{"item_code": "SKU-1", "qty": 1, "rate": 10.0}],
            notes=None,
        )
        client.submit_quotation.assert_called_once_with("QTN-0001")
        assert client.download_quotation_pdf.call_count == 2
        client.download_quotation_pdf.assert_called_with("QTN-0001")
        delete_files.assert_called_once()
        repo.update_quote_fields.assert_any_call(reply.id, clear_attachments_json=True)


def test_fulfill_fails_when_customer_missing_and_creation_disabled() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    reply = _quote_reply()
    integration_config = {"allow_create_quotation": True, "allow_create_customer": False}
    client = MagicMock()
    client.find_customer.return_value = None

    with patch(
        "evenor.application.quote_fulfillment_service.erpnext_integration_for_tenant",
        return_value=(client, integration_config),
    ), patch(
        "evenor.application.quote_fulfillment_service.ensure_erpnext_customer",
        side_effect=CustomerProvisioningError("Customer creation is disabled for this connector"),
    ):
        with pytest.raises(QuoteFulfillmentError, match="Customer creation is disabled"):
            svc.fulfill_and_approve(reply, config={})


def test_fulfill_precreated_quote_submits_and_redownloads_without_recreating() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    base = _quote_reply()
    reply = replace(
        base,
        quote_external_id="QTN-0001",
        attachments_json=json.dumps(
            [{"filename": "QTN-0001.pdf", "path": "/tmp/QTN-0001.pdf", "mime_type": "application/pdf"}]
        ),
    )
    integration_config = {"allow_create_quotation": True, "allow_create_customer": True}
    client = MagicMock()
    client.submit_quotation.return_value = None
    client.download_quotation_pdf.return_value = b"%PDF-new"

    with patch(
        "evenor.application.quote_fulfillment_service.erpnext_integration_for_tenant",
        return_value=(client, integration_config),
    ), patch(
        "evenor.application.quote_fulfillment_service.quotation_erp_modified",
        return_value="2026-06-15 14:17:39",
    ), patch(
        "evenor.application.quote_fulfillment_service.approve_pending_reply",
        return_value=reply,
    ) as approve_mock, patch(
        "evenor.application.quote_fulfillment_service.SqlAlchemyPendingReplyRepository"
    ) as repo_cls, patch(
        "evenor.application.quote_fulfillment_service.delete_attachment_files",
    ) as delete_files, patch(
        "evenor.application.quote_fulfillment_service.store_quote_pdf",
        return_value=settings.data_root / "quotes" / "bot" / "QTN-0001.pdf",
    ):
        repo = repo_cls.return_value
        svc.fulfill_and_approve(reply, config={})
        client.create_quotation.assert_not_called()
        client.submit_quotation.assert_called_once_with("QTN-0001")
        client.download_quotation_pdf.assert_called_once_with("QTN-0001")
        delete_files.assert_called()
        approve_mock.assert_called_once()
        repo.update_quote_fields.assert_any_call(reply.id, clear_attachments_json=True)


def test_fulfill_aborts_when_submit_fails() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    reply = replace(_quote_reply(), quote_external_id="QTN-0001")
    integration_config = {"allow_create_quotation": True, "allow_create_customer": True}
    client = MagicMock()
    client.submit_quotation.return_value = "Submit failed"

    with patch(
        "evenor.application.quote_fulfillment_service.erpnext_integration_for_tenant",
        return_value=(client, integration_config),
    ), patch(
        "evenor.application.quote_fulfillment_service.quotation_erp_modified",
        return_value="2026-06-15 14:17:39",
    ), patch(
        "evenor.application.quote_fulfillment_service.approve_pending_reply",
    ) as approve_mock, patch(
        "evenor.application.quote_fulfillment_service.SqlAlchemyPendingReplyRepository"
    ) as repo_cls:
        repo = repo_cls.return_value
        with pytest.raises(QuoteFulfillmentError, match="Submit failed"):
            svc.fulfill_and_approve(reply, config={})
        approve_mock.assert_not_called()
        repo.update_quote_fields.assert_called_with(reply.id, fulfillment_error="Submit failed")


def test_refresh_quote_pdf_returns_attachments_and_modified(test_settings) -> None:
    from evenor.application.quote_fulfillment_service import refresh_quote_pdf

    session = MagicMock()
    client = MagicMock()
    client.download_quotation_pdf.return_value = b"%PDF"
    client.get_quotation.return_value = {"modified": "2026-06-15 14:18:15"}

    with patch(
        "evenor.application.quote_fulfillment_service.erpnext_integration_for_tenant",
        return_value=(client, {}),
    ), patch(
        "evenor.application.quote_fulfillment_service.store_quote_pdf",
        return_value=test_settings.data_root / "quotes" / "bot" / "QTN-0001.pdf",
    ), patch(
        "evenor.application.quote_fulfillment_service.quotation_erp_modified",
        return_value="2026-06-15 14:18:15",
    ):
        attachments_json, erp_modified = refresh_quote_pdf(
            session,
            tenant_id=1,
            settings=test_settings,
            tenant_slug="bot",
            quote_name="QTN-0001",
        )

    assert "QTN-0001.pdf" in attachments_json
    assert erp_modified == "2026-06-15 14:18:15"


def test_retry_quote_fulfillment_creates_when_missing() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    reply = _quote_reply()
    created = MagicMock(
        quote_name="QTN-0042",
        attachments_json='[{"filename":"QTN-0042.pdf","path":"/tmp/q.pdf"}]',
        resolved_json=reply.quote_resolved_json,
        quote_erp_modified="2026-06-15 14:17:39",
    )

    with patch(
        "evenor.application.quote_fulfillment_service.create_quote_for_pending_reply",
        return_value=created,
    ), patch(
        "evenor.application.quote_fulfillment_service._merge_manual_and_quote_attachments",
        return_value=created.attachments_json,
    ) as merge_mock, patch(
        "evenor.application.quote_fulfillment_service.SqlAlchemyPendingReplyRepository"
    ) as repo_cls:
        repo = repo_cls.return_value
        svc.retry_quote_fulfillment(reply)

    merge_mock.assert_called_once()
    repo.update_quote_fields.assert_called_once_with(
        reply.id,
        quote_external_id="QTN-0042",
        quote_resolved_json=reply.quote_resolved_json,
        attachments_json=created.attachments_json,
        fulfillment_error=None,
        quote_erp_modified="2026-06-15 14:17:39",
    )


def test_retry_quote_fulfillment_refreshes_pdf_when_quote_exists() -> None:
    session = MagicMock()
    settings = get_settings()
    svc = QuoteFulfillmentService(session, settings=settings, tenant_slug="bot")
    reply = replace(_quote_reply(), quote_external_id="QTN-0001")

    with patch(
        "evenor.application.quote_fulfillment_service.refresh_quote_pdf",
        return_value=('[{"filename":"QTN-0001.pdf","path":"/tmp/q.pdf"}]', "2026-06-15 14:18:15"),
    ), patch(
        "evenor.application.quote_fulfillment_service.SqlAlchemyPendingReplyRepository"
    ) as repo_cls:
        repo = repo_cls.return_value
        svc.retry_quote_fulfillment(reply)

    repo.update_quote_fields.assert_called_once_with(
        reply.id,
        attachments_json='[{"filename":"QTN-0001.pdf","path":"/tmp/q.pdf"}]',
        fulfillment_error=None,
        quote_erp_modified="2026-06-15 14:18:15",
    )
