from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from chatbot.automation.modules.base import FulfillmentMode, ParsedHook
from chatbot.domain.models.hook import HookEvent
from chatbot.domain.models.order import OrderAction, OrderCommand, OrderItem

_ORDERS_PROMPT = """Order management (local database, processed by automation worker):
- "order.create", "order.update", "order.delete"
- Payload fields: action (optional if type is set), name, tel, address, pin, products [{qty, product}], reason
Example:
{"type":"order.create","tel":"23057770000","products":[{"qty":1,"product":"Diffuser X"}]}"""


def parse_order_command(payload: dict[str, Any]) -> OrderCommand | None:
    action_raw = payload.get("action")
    if not action_raw and isinstance(payload.get("type"), str):
        parts = payload["type"].split(".")
        action_raw = parts[-1] if parts else None
    try:
        action = OrderAction(str(action_raw).strip().lower())
    except (TypeError, ValueError):
        return None
    products = payload.get("products")
    items: list[OrderItem] = []
    if isinstance(products, list):
        for raw in products:
            if not isinstance(raw, dict):
                continue
            try:
                qty = int(raw.get("qty", 0))
            except (TypeError, ValueError):
                continue
            product = str(raw.get("product", "")).strip()
            if qty > 0 and product:
                items.append(OrderItem(qty=qty, product=product))
    return OrderCommand(
        action=action,
        name=str(payload.get("name")).strip() if payload.get("name") else None,
        tel=str(payload.get("tel")).strip() if payload.get("tel") else None,
        address=str(payload.get("address")).strip() if payload.get("address") else None,
        pin=str(payload.get("pin")).strip() if payload.get("pin") else None,
        products=tuple(items),
        reason=str(payload.get("reason")).strip() if payload.get("reason") else None,
        raw_payload=payload,
    )


class CoreOrdersModule:
    id = "core.orders"
    label = "Local orders"
    description = (
        "Capture orders in the platform database when the bot emits order.create/update/delete hooks. "
        "Processed by the automation worker. Use for clients without ERP quoting."
    )
    hook_type_prefixes = ("order", "order.")
    requires_integration = None
    fulfillment_mode = FulfillmentMode.WORKER
    ui_enabled = True

    def prompt_fragment(self) -> str:
        return _ORDERS_PROMPT

    def matches(self, hook_type: str) -> bool:
        return hook_type == "order" or hook_type.startswith("order.")

    def parse(self, payload: dict[str, Any]) -> ParsedHook | None:
        command = parse_order_command(payload)
        if command is None:
            return None
        hook_type = str(payload.get("type") or "order").strip()
        return ParsedHook(hook_type=hook_type, payload=payload)

    def handle_worker(self, session: Session, hook: HookEvent) -> None:
        from chatbot.automation.handlers.order_handler import handle_order_hook

        handle_order_hook(session, hook)


def parse_order_payload_json(payload_json: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
