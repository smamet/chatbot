from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.channel_outbound import approve_pending_reply
from chatbot.application.customer_access_gate import can_create_quotation, parse_session_identity
from chatbot.application.customer_provisioning_service import (
    CustomerProvisioningError,
    ensure_erpnext_customer,
)
from chatbot.application.outbound_orchestrator import erpnext_integration_for_tenant
from chatbot.application.product_resolver import LineMatchStatus, resolved_lines_from_json
from chatbot.application.quote_pdf_storage import (
    attachment_entry,
    delete_attachment_files,
    encode_attachments_json,
    load_attachments_from_json,
    quote_pdf_dashboard_url,
    safe_quote_filename,
    store_quote_pdf,
)
from chatbot.automation.modules.erpnext.quote import QuoteProposal, parse_quote_proposal
from chatbot.config.settings import Settings
from chatbot.domain.contracts.llm_client import LlmResult
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.outbound_attachment import OutboundAttachment
from chatbot.domain.models.pending_reply import PendingReply


class QuoteFulfillmentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QuoteCreateResult:
    quote_name: str
    pdf_bytes: bytes | None
    pdf_path: Path | None
    pdf_filename: str | None
    pdf_url: str | None
    pdf_warning: str | None
    attachments_json: str | None
    resolved_json: str


def all_lines_resolved(resolved_json: str | None) -> bool:
    lines = resolved_lines_from_json(resolved_json)
    if not lines:
        return False
    return all(
        line.get("status") == LineMatchStatus.RESOLVED.value and line.get("item_code")
        for line in lines
    )


def resolve_quote_hook(
    session: Session,
    tenant_id: int,
    result: LlmResult,
) -> tuple[QuoteProposal, str] | None:
    hook_type = getattr(result, "hook_type", None)
    payload_raw = getattr(result, "hook_payload_json", None)
    if hook_type != "quote.create" or not payload_raw:
        return None
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    proposal = parse_quote_proposal(payload)
    if proposal is None:
        return None
    integration = erpnext_integration_for_tenant(session, tenant_id)
    if integration is None:
        return None
    client, _config = integration
    from chatbot.application.product_resolver import ProductResolver, resolved_lines_to_json

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
    return proposal, resolved_json


def create_quote_and_pdf(
    client: ErpNextClient,
    config: dict[str, Any],
    settings: Settings,
    tenant_slug: str,
    *,
    customer: str,
    lines: list[dict[str, Any]],
    notes: str | None = None,
    ttl_seconds: int | None = None,
) -> tuple[str, bytes | None, Path | None, str | None]:
    created = client.create_quotation(customer, lines, notes=notes)
    quote_name = str(created.get("name", "")).strip()
    if not quote_name:
        raise QuoteFulfillmentError("ERPNext did not return a quotation name")

    pdf_bytes = client.download_quotation_pdf(quote_name)
    pdf_path: Path | None = None
    pdf_warning: str | None = None
    if pdf_bytes:
        pdf_path = store_quote_pdf(
            settings,
            tenant_slug,
            quote_name,
            pdf_bytes,
            ttl_seconds=ttl_seconds,
        )
    else:
        detail = client.last_pdf_error or "PDF could not be downloaded from ERPNext."
        pdf_warning = f"PDF could not be downloaded from ERPNext ({detail})."
    return quote_name, pdf_bytes, pdf_path, pdf_warning


def create_quote_for_session(
    session: Session,
    *,
    tenant_id: int,
    settings: Settings,
    tenant_slug: str,
    session_id: str,
    proposal: QuoteProposal,
    resolved_json: str,
    ttl_seconds: int | None = None,
    company_name: str | None = None,
) -> QuoteCreateResult:
    integration = erpnext_integration_for_tenant(session, tenant_id)
    if integration is None:
        raise QuoteFulfillmentError("ERPNext integration is not configured or inactive")
    client, integration_config = integration
    if not can_create_quotation(integration_config):
        raise QuoteFulfillmentError("Quotation creation is disabled for this connector")
    if not all_lines_resolved(resolved_json):
        raise QuoteFulfillmentError("Not all quote lines are resolved")

    lines = resolved_lines_from_json(resolved_json)
    quote_lines = [
        {
            "item_code": str(line["item_code"]),
            "qty": int(line["qty"]),
            "rate": line.get("rate"),
        }
        for line in lines
    ]
    customer = _resolve_customer_for_session(session_id, client)
    if not customer:
        email, phone = parse_session_identity(session_id)
        try:
            customer = ensure_erpnext_customer(
                client,
                integration_config,
                email=email,
                phone=phone,
                company_name=company_name,
            )
        except CustomerProvisioningError as exc:
            raise QuoteFulfillmentError(str(exc)) from exc

    quote_name, pdf_bytes, pdf_path, pdf_warning = create_quote_and_pdf(
        client,
        integration_config,
        settings,
        tenant_slug,
        customer=customer,
        lines=quote_lines,
        notes=proposal.notes,
        ttl_seconds=ttl_seconds,
    )
    pdf_filename = f"{safe_quote_filename(quote_name)}.pdf" if pdf_bytes else None
    pdf_url = quote_pdf_dashboard_url(tenant_slug, quote_name) if pdf_bytes else None
    attachments_json: str | None = None
    if pdf_path is not None and pdf_filename:
        attachments_json = encode_attachments_json(
            [attachment_entry(path=pdf_path, filename=pdf_filename)]
        )
    return QuoteCreateResult(
        quote_name=quote_name,
        pdf_bytes=pdf_bytes,
        pdf_path=pdf_path,
        pdf_filename=pdf_filename,
        pdf_url=pdf_url,
        pdf_warning=pdf_warning,
        attachments_json=attachments_json,
        resolved_json=resolved_json,
    )


class QuoteFulfillmentService:
    def __init__(self, session: Session, *, settings: Settings, tenant_slug: str) -> None:
        self._session = session
        self._settings = settings
        self._tenant_slug = tenant_slug

    def fulfill_and_approve(
        self,
        reply: PendingReply,
        *,
        config: dict,
        quote_resolved_json: str | None = None,
    ) -> PendingReply:
        if reply.fulfillment_kind != FulfillmentKind.ERPNEXT_QUOTE:
            approved = approve_pending_reply(
                self._session,
                reply,
                config=config,
                settings=self._settings,
            )
            if approved is None:
                raise QuoteFulfillmentError("Failed to approve reply")
            return approved

        repo = SqlAlchemyPendingReplyRepository(self._session)
        resolved_raw = quote_resolved_json or reply.quote_resolved_json
        lines = resolved_lines_from_json(resolved_raw)
        if not lines:
            raise QuoteFulfillmentError("No quote lines to fulfill")
        for line in lines:
            if line.get("status") != LineMatchStatus.RESOLVED.value:
                raise QuoteFulfillmentError(
                    f"Unresolved product line: {line.get('requested_label', '?')}"
                )
            if not line.get("item_code"):
                raise QuoteFulfillmentError(
                    f"Missing item_code for line: {line.get('requested_label', '?')}"
                )

        integration = erpnext_integration_for_tenant(self._session, reply.tenant_id)
        if integration is None:
            raise QuoteFulfillmentError("ERPNext integration is not configured or inactive")
        client, integration_config = integration

        if not can_create_quotation(integration_config):
            raise QuoteFulfillmentError("Quotation creation is disabled for this connector")

        attachments: list[OutboundAttachment] = []
        attachments_json: str | None = reply.attachments_json
        quote_name = (reply.quote_external_id or "").strip()

        if quote_name and attachments_json:
            attachments = load_attachments_from_json(attachments_json)
            if not attachments:
                raise QuoteFulfillmentError("Pre-created quote PDF is missing on disk")
        else:
            customer = self._resolve_customer(reply, client)
            if not customer:
                email, phone = parse_session_identity(reply.session_id)
                try:
                    customer = ensure_erpnext_customer(
                        client,
                        integration_config,
                        email=email,
                        phone=phone,
                    )
                except CustomerProvisioningError as exc:
                    raise QuoteFulfillmentError(str(exc)) from exc

            quote_lines = [
                {
                    "item_code": str(line["item_code"]),
                    "qty": int(line["qty"]),
                    "rate": line.get("rate"),
                }
                for line in lines
            ]
            notes = _quote_notes_from_reply(reply)
            quote_name, pdf_bytes, pdf_path, _pdf_warning = create_quote_and_pdf(
                client,
                integration_config,
                self._settings,
                self._tenant_slug,
                customer=customer,
                lines=quote_lines,
                notes=notes,
            )
            attachments_json = None
            if pdf_bytes and pdf_path is not None:
                pdf_filename = f"{safe_quote_filename(quote_name)}.pdf"
                attachments.append(
                    OutboundAttachment(
                        filename=pdf_filename,
                        data=pdf_bytes,
                        mime_type="application/pdf",
                    )
                )
                attachments_json = encode_attachments_json(
                    [attachment_entry(path=pdf_path, filename=pdf_filename)]
                )
            repo.update_quote_fields(
                reply.id,
                quote_resolved_json=resolved_raw,
                quote_external_id=quote_name,
                attachments_json=attachments_json,
                fulfillment_error=None,
            )

        try:
            approved = approve_pending_reply(
                self._session,
                reply,
                config=config,
                settings=self._settings,
                attachments=attachments or None,
            )
        except Exception as exc:
            repo.update_quote_fields(
                reply.id,
                fulfillment_error=str(exc),
            )
            raise QuoteFulfillmentError(str(exc)) from exc
        if approved is None:
            raise QuoteFulfillmentError("Failed to approve reply")
        if attachments_json:
            delete_attachment_files(attachments_json)
            repo.update_quote_fields(reply.id, clear_attachments_json=True)
        return approved

    def _resolve_customer(self, reply: PendingReply, client: ErpNextClient) -> str | None:
        return _resolve_customer_for_session(reply.session_id, client)


def _resolve_customer_for_session(session_id: str, client: ErpNextClient) -> str | None:
    email, phone = parse_session_identity(session_id)
    return client.find_customer(email=email, phone=phone)


def _quote_notes_from_reply(reply: PendingReply) -> str | None:
    if not reply.quote_proposal_json:
        return None
    try:
        proposal = json.loads(reply.quote_proposal_json)
    except json.JSONDecodeError:
        return None
    if isinstance(proposal, dict) and proposal.get("notes"):
        return str(proposal["notes"])
    return None
