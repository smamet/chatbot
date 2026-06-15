from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.application.customer_access_gate import can_create_quotation
from chatbot.application.integration_service import IntegrationService
from chatbot.application.product_resolver import ProductResolver, resolved_lines_to_json
from chatbot.automation.modules.base import FulfillmentMode
from chatbot.automation.modules.erpnext.quote import parse_quote_proposal
from chatbot.automation.modules.registry import module_for_hook_type
from chatbot.config.settings import Settings
from chatbot.domain.contracts.llm_client import LlmResult
from chatbot.domain.models.connector import Connector, ConnectorMode, ConnectorType
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.integration import IntegrationType
from chatbot.domain.models.pending_reply import PendingReply


def _erpnext_client_for_tenant(session: Session, tenant_id: int) -> ErpNextClient | None:
    pair = erpnext_integration_for_tenant(session, tenant_id)
    return pair[0] if pair else None


def erpnext_integration_for_tenant(
    session: Session, tenant_id: int
) -> tuple[ErpNextClient, dict[str, Any]] | None:
    integ_svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = integ_svc.find_active(tenant_id, type=IntegrationType.ERPNEXT)
    if not integration:
        return None
    return ErpNextClient(integration.config), integration.config


def _queue_quote_pending(
    session: Session,
    *,
    tenant_id: int,
    connector: Connector,
    session_id: str,
    recipient_id: str,
    result: LlmResult,
    resolved_json: str | None,
    quote_external_id: str | None = None,
    attachments_json: str | None = None,
) -> PendingReply:
    from chatbot.application.channel_outbound import queue_pending_reply

    pending = queue_pending_reply(
        session,
        tenant_id=tenant_id,
        connector_id=connector.id,
        session_id=session_id,
        channel=connector.type.value,
        recipient_id=recipient_id,
        draft_text=result.text,
        hook_event_id=getattr(result, "hook_event_id", None),
        fulfillment_kind=FulfillmentKind.ERPNEXT_QUOTE,
        quote_proposal_json=getattr(result, "hook_payload_json", None),
        quote_resolved_json=resolved_json,
        quote_external_id=quote_external_id,
        attachments_json=attachments_json,
    )
    return pending


def queue_after_chat(
    session: Session,
    *,
    tenant_id: int,
    connector: Connector,
    session_id: str,
    recipient_id: str,
    result: LlmResult,
    settings: Settings,
    tenant_slug: str,
) -> tuple[str, PendingReply | None]:
    """Return (status, pending_reply_if_queued)."""
    from chatbot.application.channel_outbound import (
        dispatch_channel_reply,
        queue_pending_reply,
        should_queue_for_validation,
    )

    hook_type = getattr(result, "hook_type", None)
    mod = module_for_hook_type(hook_type) if hook_type else None
    is_quote = mod is not None and mod.fulfillment_mode == FulfillmentMode.VALIDATION

    if is_quote and getattr(result, "hook_payload_json", None):
        payload = json.loads(result.hook_payload_json)
        proposal = parse_quote_proposal(payload) if isinstance(payload, dict) else None
        if proposal is None:
            is_quote = False
        else:
            from chatbot.application.quote_fulfillment_service import (
                all_lines_resolved,
                create_quote_for_session,
            )

            client = _erpnext_client_for_tenant(session, tenant_id)
            resolved_json = None
            if client:
                resolver = ProductResolver(client)
                lines = [
                    {
                        "product": line.product,
                        "qty": line.qty,
                        "item_code": line.item_code,
                    }
                    for line in proposal.lines
                ]
                resolved_json = resolved_lines_to_json(resolver.resolve_all(lines))

            integration = erpnext_integration_for_tenant(session, tenant_id)
            integration_config = integration[1] if integration else {}
            quote_external_id: str | None = None
            attachments_json: str | None = None
            quote_erp_modified: str | None = None
            if (
                integration
                and can_create_quotation(integration_config)
                and all_lines_resolved(resolved_json)
            ):
                try:
                    created = create_quote_for_session(
                        session,
                        tenant_id=tenant_id,
                        settings=settings,
                        tenant_slug=tenant_slug,
                        session_id=session_id,
                        proposal=proposal,
                        resolved_json=resolved_json or "[]",
                    )
                    quote_external_id = created.quote_name
                    attachments_json = created.attachments_json
                    resolved_json = created.resolved_json
                    quote_erp_modified = created.quote_erp_modified
                except Exception as exc:
                    repo = SqlAlchemyPendingReplyRepository(session)
                    pending = _queue_quote_pending(
                        session,
                        tenant_id=tenant_id,
                        connector=connector,
                        session_id=session_id,
                        recipient_id=recipient_id,
                        result=result,
                        resolved_json=resolved_json,
                    )
                    repo.update_quote_fields(pending.id, fulfillment_error=str(exc))
                    return "queued", pending

            pending = _queue_quote_pending(
                session,
                tenant_id=tenant_id,
                connector=connector,
                session_id=session_id,
                recipient_id=recipient_id,
                result=result,
                resolved_json=resolved_json,
                quote_external_id=quote_external_id,
                attachments_json=attachments_json,
            )
            if quote_external_id and quote_erp_modified:
                SqlAlchemyPendingReplyRepository(session).update_quote_fields(
                    pending.id,
                    quote_erp_modified=quote_erp_modified,
                )
            return "queued", pending

    if should_queue_for_validation(connector):
        pending = queue_pending_reply(
            session,
            tenant_id=tenant_id,
            connector_id=connector.id,
            session_id=session_id,
            channel=connector.type.value,
            recipient_id=recipient_id,
            draft_text=result.text,
            hook_event_id=getattr(result, "hook_event_id", None),
        )
        return "queued", pending

    dispatch_channel_reply(
        channel=connector.type.value,
        recipient_id=recipient_id,
        text=result.text,
        config=connector.config,
        settings=settings,
    )
    return "ok", None


def get_outbound_connector_for_channel(
    connectors: ConnectorService,
    tenant_id: int,
    channel: ConnectorType,
) -> Connector | None:
    from chatbot.application.channel_outbound import get_outbound_connector

    return get_outbound_connector(connectors, tenant_id, channel)
