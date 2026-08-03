"""UI helpers for trader LLM decision summaries (live + backtest reports)."""

from __future__ import annotations

import re
from typing import Any

_ID_IN_TEXT = re.compile(r"\b[op]\d+\b", re.IGNORECASE)


def _format_level(level: Any) -> str:
    if level is None or level == "":
        return ""
    try:
        n = float(level)
    except (TypeError, ValueError):
        return str(level)
    if abs(n) < 50:
        text = f"{n:.5f}".rstrip("0").rstrip(".")
    else:
        text = f"{n:.2f}".rstrip("0").rstrip(".")
    return text


def _with_level(label: str, level: Any, *, arrow: bool = False) -> str:
    level_s = _format_level(level)
    if not level_s:
        return label
    if arrow:
        return f"{label} → {level_s}"
    return f"{label} @ {level_s}"


def _normalize_purpose(purpose: Any) -> str:
    return str(purpose or "").strip().lower()


def _normalize_order_type(order_type: Any) -> str:
    return str(order_type or "").strip().upper()


def _cancel_target(purpose: str, order_type: str = "") -> str:
    """Human target for a cancel chip, e.g. 'hedge', 'TP', 'entry (limit)'."""
    p = _normalize_purpose(purpose)
    t = _normalize_order_type(order_type)
    if p in ("hedge_cover", "hedge"):
        return "hedge"
    if p == "tp":
        return "TP"
    if p == "entry":
        if t == "STOP":
            return "entry (stop)"
        return "entry (limit)"
    if p == "close":
        return "close"
    return ""


def _working_order_lookup(
    working_orders: list[Any] | dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Map order id → {purpose, type} from a snapshot working-order list/dict."""
    out: dict[str, dict[str, Any]] = {}
    if isinstance(working_orders, dict):
        items = working_orders.values()
    elif isinstance(working_orders, list):
        items = working_orders
    else:
        return out
    for raw in items:
        if not isinstance(raw, dict):
            continue
        oid = str(raw.get("id") or "").strip()
        if not oid:
            continue
        out[oid] = {
            "purpose": _normalize_purpose(raw.get("purpose")),
            "type": _normalize_order_type(raw.get("type")),
        }
    return out


def summarize_llm_action(
    action: dict[str, Any],
    *,
    order_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Human label for one LLM action (op + purpose + optional level)."""
    op = str(action.get("op") or "").strip()
    purpose = _normalize_purpose(action.get("purpose"))
    level = action.get("level")
    order_id = str(action.get("order_id") or "").strip()
    order_type = _normalize_order_type(action.get("type"))

    if op == "market_open":
        return "Market open"
    if op == "market_close":
        return "Market close"

    if op == "amend_order":
        if purpose == "tp":
            return _with_level("TP change", level, arrow=True)
        if purpose == "entry":
            return _with_level("Entry change", level, arrow=True)
        if purpose in ("hedge_cover", "hedge"):
            return _with_level("Hedge change", level, arrow=True)
        base = f"Amend {purpose}" if purpose else "Amend order"
        return _with_level(base, level, arrow=True)

    if op == "cancel_order":
        if order_lookup and order_id:
            meta = order_lookup.get(order_id) or {}
            purpose = purpose or _normalize_purpose(meta.get("purpose"))
            order_type = order_type or _normalize_order_type(meta.get("type"))
        target = _cancel_target(purpose, order_type)
        if target:
            return f"Cancel {target}"
        if order_id:
            return f"Cancel {order_id}"
        return "Cancel order"

    if op in ("place_limit", "place_stop"):
        kind = "stop" if op == "place_stop" else "place"
        if purpose == "entry":
            return _with_level(f"Entry {kind}", level)
        if purpose == "tp":
            return _with_level("TP place", level)
        if purpose in ("hedge_cover", "hedge"):
            return _with_level("Hedge", level)
        if purpose == "close":
            return _with_level("Close place", level)
        base = f"{purpose} {kind}" if purpose else f"Order {kind}"
        return _with_level(base, level)

    if op:
        return op.replace("_", " ")
    return "Action"


def summarize_llm_actions(
    actions: list[Any] | None,
    *,
    working_orders: list[Any] | dict[str, Any] | None = None,
) -> list[str]:
    """Return human-readable chips for a decision's actions (or Hold if empty)."""
    if not actions:
        return ["Hold"]
    lookup = _working_order_lookup(working_orders)
    chips: list[str] = []
    for raw in actions:
        if not isinstance(raw, dict):
            continue
        label = summarize_llm_action(raw, order_lookup=lookup)
        if label:
            chips.append(label)
    return chips or ["Hold"]


def _add_id(ids: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    if re.fullmatch(r"[op]\d+", text, flags=re.IGNORECASE):
        ids.add(text.lower())
        return
    for match in _ID_IN_TEXT.findall(text):
        ids.add(match.lower())


def _ids_from_book_items(items: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(items, dict):
        for key, raw in items.items():
            _add_id(ids, key)
            if isinstance(raw, dict):
                _add_id(ids, raw.get("id"))
    elif isinstance(items, list):
        for raw in items:
            if isinstance(raw, dict):
                _add_id(ids, raw.get("id"))
            else:
                _add_id(ids, raw)
    return ids


def decision_search_ids(
    *,
    actions: list[Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    executed: list[Any] | None = None,
    rejected: list[Any] | None = None,
) -> list[str]:
    """Collect local order/position ids (oN / pN) for cycle search."""
    ids: set[str] = set()
    for raw in actions or []:
        if not isinstance(raw, dict):
            continue
        _add_id(ids, raw.get("order_id"))
        _add_id(ids, raw.get("position_id"))
    snap = snapshot if isinstance(snapshot, dict) else {}
    ids |= _ids_from_book_items(snap.get("working_orders"))
    ids |= _ids_from_book_items(snap.get("positions"))
    for bucket in (executed, rejected):
        for item in bucket or []:
            _add_id(ids, item)
    return sorted(ids)
