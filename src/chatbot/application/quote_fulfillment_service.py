from __future__ import annotations

import json

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
    safe_quote_filename,
    store_quote_pdf,
)
from chatbot.config.settings import Settings
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.outbound_attachment import OutboundAttachment
from chatbot.domain.models.pending_reply import PendingReply


class QuoteFulfillmentError(RuntimeError):
    pass


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
        notes = None
        if reply.quote_proposal_json:
            try:
                proposal = json.loads(reply.quote_proposal_json)
                if isinstance(proposal, dict) and proposal.get("notes"):
                    notes = str(proposal["notes"])
            except json.JSONDecodeError:
                pass

        created = client.create_quotation(customer, quote_lines, notes=notes)
        quote_name = str(created.get("name", "")).strip()
        if not quote_name:
            raise QuoteFulfillmentError("ERPNext did not return a quotation name")

        pdf_bytes = client.download_quotation_pdf(quote_name)
        attachments: list[OutboundAttachment] = []
        attachments_json: str | None = None
        if pdf_bytes:
            pdf_path = store_quote_pdf(self._settings, self._tenant_slug, quote_name, pdf_bytes)
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
        email, phone = parse_session_identity(reply.session_id)
        return client.find_customer(email=email, phone=phone)


def _session_email(session_id: str) -> str | None:
    if session_id.startswith("email:"):
        return session_id.split(":", 1)[1].strip() or None
    return None


def _session_phone(session_id: str) -> str | None:
    if session_id.startswith("whatsapp:"):
        return session_id.split(":", 1)[1].strip() or None
    return None
