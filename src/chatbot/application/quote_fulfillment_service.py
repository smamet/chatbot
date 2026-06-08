from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.channel_outbound import approve_pending_reply
from chatbot.application.outbound_orchestrator import _erpnext_client_for_tenant
from chatbot.application.product_resolver import LineMatchStatus, resolved_lines_from_json
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

        client = _erpnext_client_for_tenant(self._session, reply.tenant_id)
        if client is None:
            raise QuoteFulfillmentError("ERPNext integration is not configured or inactive")

        customer = self._resolve_customer(reply, client)
        if not customer:
            raise QuoteFulfillmentError(
                "Customer not found in ERPNext for this session (email/WhatsApp identity required)"
            )

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
        if pdf_bytes:
            pdf_path = self._store_pdf(quote_name, pdf_bytes)
            attachments.append(
                OutboundAttachment(
                    filename=f"{quote_name}.pdf",
                    data=pdf_bytes,
                    mime_type="application/pdf",
                )
            )
            attachments_json = json.dumps(
                [{"filename": f"{quote_name}.pdf", "path": str(pdf_path), "mime_type": "application/pdf"}]
            )
        else:
            attachments_json = None

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
        return approved

    def _resolve_customer(self, reply: PendingReply, client: ErpNextClient) -> str | None:
        from chatbot.application.customer_access_gate import parse_session_identity

        email, phone = parse_session_identity(reply.session_id)
        return client.find_customer(email=email, phone=phone)

    def _store_pdf(self, quote_name: str, pdf_bytes: bytes) -> Path:
        root = self._settings.data_root / "quotes" / self._tenant_slug
        root.mkdir(parents=True, exist_ok=True)
        safe = quote_name.replace("/", "-")
        path = root / f"{safe}.pdf"
        path.write_bytes(pdf_bytes)
        return path


def _session_email(session_id: str) -> str | None:
    if session_id.startswith("email:"):
        return session_id.split(":", 1)[1].strip() or None
    return None


def _session_phone(session_id: str) -> str | None:
    if session_id.startswith("whatsapp:"):
        return session_id.split(":", 1)[1].strip() or None
    return None
