from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from evenor.application.outbound_orchestrator import queue_after_chat
from evenor.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from evenor.domain.models.fulfillment import FulfillmentKind


def _email_connector():
    return SimpleNamespace(
        id=1,
        type=ConnectorType.EMAIL,
        direction=ConnectorDirection.OUT,
        mode=ConnectorMode.VALIDATION,
        active=True,
        config={"from_addr": "bot@example.com"},
    )


def test_queue_after_chat_precreates_quote_when_enabled() -> None:
    session = MagicMock()
    result = SimpleNamespace(
        text="Here is your quote",
        hook_type="quote.create",
        hook_payload_json='{"type":"quote.create","lines":[{"product":"Widget","qty":1}]}',
        hook_event_id=7,
    )
    created = SimpleNamespace(
        quote_name="QTN-0001",
        attachments_json='[{"filename":"QTN-0001.pdf","path":"/tmp/q.pdf"}]',
        resolved_json='[{"requested_label":"Widget","qty":1,"item_code":"SKU-1","status":"resolved"}]',
        quote_erp_modified="2026-06-15 14:17:39",
    )
    pending = SimpleNamespace(id=99)

    with patch(
        "evenor.application.outbound_orchestrator._erpnext_client_for_tenant",
        return_value=MagicMock(),
    ), patch(
        "evenor.application.outbound_orchestrator.resolved_lines_to_json",
        return_value=created.resolved_json,
    ), patch(
        "evenor.application.outbound_orchestrator.erpnext_integration_for_tenant",
        return_value=(MagicMock(), {"allow_create_quotation": True}),
    ), patch(
        "evenor.application.quote_fulfillment_service.create_quote_for_session",
        return_value=created,
    ), patch(
        "evenor.application.outbound_orchestrator._queue_quote_pending",
        return_value=pending,
    ) as queue_mock, patch(
        "evenor.application.outbound_orchestrator.SqlAlchemyPendingReplyRepository",
    ) as repo_cls:
        status, out = queue_after_chat(
            session,
            tenant_id=1,
            connector=_email_connector(),
            session_id="email:a@example.com",
            recipient_id="a@example.com",
            result=result,
            settings=MagicMock(),
            tenant_slug="bot",
        )

    assert status == "queued"
    assert out is pending
    queue_mock.assert_called_once()
    assert queue_mock.call_args.kwargs["quote_external_id"] == "QTN-0001"
    assert queue_mock.call_args.kwargs["attachments_json"] == created.attachments_json
    repo_cls.return_value.update_quote_fields.assert_called_once_with(
        pending.id,
        quote_erp_modified="2026-06-15 14:17:39",
    )


def test_queue_after_chat_keeps_manual_validation_when_creation_disabled() -> None:
    session = MagicMock()
    result = SimpleNamespace(
        text="Here is your quote",
        hook_type="quote.create",
        hook_payload_json='{"type":"quote.create","lines":[{"product":"Widget","qty":1}]}',
        hook_event_id=None,
    )
    pending = SimpleNamespace(id=1)

    with patch(
        "evenor.application.outbound_orchestrator._erpnext_client_for_tenant",
        return_value=MagicMock(),
    ), patch(
        "evenor.application.outbound_orchestrator.resolved_lines_to_json",
        return_value='[{"requested_label":"Widget","qty":1,"item_code":"SKU-1","status":"resolved"}]',
    ), patch(
        "evenor.application.outbound_orchestrator.erpnext_integration_for_tenant",
        return_value=(MagicMock(), {"allow_create_quotation": False}),
    ), patch(
        "evenor.application.outbound_orchestrator._queue_quote_pending",
        return_value=pending,
    ) as queue_mock:
        status, out = queue_after_chat(
            session,
            tenant_id=1,
            connector=_email_connector(),
            session_id="email:a@example.com",
            recipient_id="a@example.com",
            result=result,
            settings=MagicMock(),
            tenant_slug="bot",
        )

    assert status == "queued"
    assert out is pending
    assert queue_mock.call_args.kwargs["quote_external_id"] is None
