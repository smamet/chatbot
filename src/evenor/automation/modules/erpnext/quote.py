from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evenor.automation.modules.base import FulfillmentMode, ParsedHook

_QUOTE_PROMPT = """ERPNext quotation (queued for human validation; created in ERPNext on approve):
- "quote.create" only when the customer confirmed products and quantities
- If any product is ambiguous, ask the customer to clarify first — do not emit the hook
- Payload: lines [{product, item_code (optional), qty}], notes (optional)
Example:
{"type":"quote.create","lines":[{"product":"Sigma 3000","qty":2}],"notes":"Delivery requested"}"""


@dataclass(frozen=True, slots=True)
class QuoteLineProposal:
    product: str
    qty: int
    item_code: str | None = None


@dataclass(frozen=True, slots=True)
class QuoteProposal:
    lines: tuple[QuoteLineProposal, ...]
    notes: str | None = None


class ErpNextQuoteModule:
    id = "erpnext.quote"
    label = "ERPNext quotes"
    description = (
        "When the bot emits quote.create, queue the reply for validation. On approve, creates a draft "
        "Quotation in ERPNext and sends the PDF to the customer. Requires an active ERPNext integration."
    )
    hook_type_prefixes = ("quote.create",)
    requires_integration = "erpnext"
    fulfillment_mode = FulfillmentMode.VALIDATION
    ui_enabled = True

    def prompt_fragment(self) -> str:
        return _QUOTE_PROMPT

    def matches(self, hook_type: str) -> bool:
        return hook_type == "quote.create"

    def parse(self, payload: dict[str, Any]) -> ParsedHook | None:
        proposal = parse_quote_proposal(payload)
        if proposal is None:
            return None
        return ParsedHook(hook_type="quote.create", payload=payload)

    def handle_worker(self, session: Any, hook: Any) -> None:
        raise RuntimeError("erpnext.quote uses validation fulfillment, not worker")


def parse_quote_proposal(payload: dict[str, Any]) -> QuoteProposal | None:
    if str(payload.get("type", "")).strip() != "quote.create":
        return None
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        return None
    lines: list[QuoteLineProposal] = []
    for raw in raw_lines:
        if not isinstance(raw, dict):
            continue
        product = str(raw.get("product", "")).strip()
        try:
            qty = int(raw.get("qty", 0))
        except (TypeError, ValueError):
            continue
        if not product or qty <= 0:
            continue
        item_code = str(raw.get("item_code", "")).strip() or None
        lines.append(QuoteLineProposal(product=product, qty=qty, item_code=item_code))
    if not lines:
        return None
    notes = str(payload.get("notes")).strip() if payload.get("notes") else None
    return QuoteProposal(lines=tuple(lines), notes=notes)
