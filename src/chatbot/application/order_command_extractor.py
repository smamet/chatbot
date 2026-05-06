from __future__ import annotations

import json
from dataclasses import dataclass

from chatbot.domain.models.order import OrderAction, OrderCommand, OrderItem

ORDER_MARKER = "===JF030A==="


@dataclass(frozen=True, slots=True)
class ExtractedOrderCommand:
    clean_reply: str
    command: OrderCommand | None
    command_json: str | None


def _to_action(value: object) -> OrderAction | None:
    if not isinstance(value, str):
        return None
    try:
        return OrderAction(value.strip().lower())
    except ValueError:
        return None


def _to_items(value: object) -> tuple[OrderItem, ...]:
    if not isinstance(value, list):
        return ()
    items: list[OrderItem] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        qty_raw = raw.get("qty")
        product_raw = raw.get("product")
        try:
            qty = int(qty_raw)
        except (TypeError, ValueError):
            continue
        product = str(product_raw or "").strip()
        if qty <= 0 or not product:
            continue
        items.append(OrderItem(qty=qty, product=product))
    return tuple(items)


def extract_order_command(text: str) -> ExtractedOrderCommand:
    marker_idx = text.find(ORDER_MARKER)
    if marker_idx < 0:
        return ExtractedOrderCommand(clean_reply=text.strip(), command=None, command_json=None)

    clean_reply = text[:marker_idx].strip()
    payload_str = text[marker_idx + len(ORDER_MARKER) :].strip()
    if not payload_str:
        return ExtractedOrderCommand(clean_reply=clean_reply, command=None, command_json=None)

    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(payload_str)
    except json.JSONDecodeError:
        return ExtractedOrderCommand(clean_reply=clean_reply, command=None, command_json=None)
    if not isinstance(payload, dict):
        return ExtractedOrderCommand(clean_reply=clean_reply, command=None, command_json=None)

    action = _to_action(payload.get("action"))
    if action is None:
        return ExtractedOrderCommand(clean_reply=clean_reply, command=None, command_json=None)

    command = OrderCommand(
        action=action,
        name=str(payload.get("name")).strip() if payload.get("name") else None,
        tel=str(payload.get("tel")).strip() if payload.get("tel") else None,
        address=str(payload.get("address")).strip() if payload.get("address") else None,
        pin=str(payload.get("pin")).strip() if payload.get("pin") else None,
        products=_to_items(payload.get("products")),
        reason=str(payload.get("reason")).strip() if payload.get("reason") else None,
        raw_payload=payload,
    )
    command_json = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return ExtractedOrderCommand(clean_reply=clean_reply, command=command, command_json=command_json)
