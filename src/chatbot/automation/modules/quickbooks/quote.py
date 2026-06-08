from __future__ import annotations

from typing import Any

from chatbot.automation.modules.base import FulfillmentMode, ParsedHook

_QB_PROMPT = """QuickBooks estimates (coming soon — not active yet)."""


class QuickBooksQuoteModule:
    id = "quickbooks.quote"
    label = "QuickBooks quotes"
    description = "Coming soon. Creates QuickBooks Estimates on approve. This checkbox is disabled until the feature is released."
    hook_type_prefixes = ("quote.create",)
    requires_integration = "quickbooks"
    fulfillment_mode = FulfillmentMode.VALIDATION
    ui_enabled = False

    def prompt_fragment(self) -> str:
        return _QB_PROMPT

    def matches(self, hook_type: str) -> bool:
        return hook_type == "quote.create"

    def parse(self, payload: dict[str, Any]) -> ParsedHook | None:
        return None

    def handle_worker(self, session: Any, hook: Any) -> None:
        raise NotImplementedError("quickbooks.quote is not implemented yet")
