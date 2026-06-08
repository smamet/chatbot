from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from chatbot.domain.contracts.customer_data_client import CustomerDataClient

logger = logging.getLogger(__name__)

_CONTACT_FIELD_ORDER = (
    "full_name",
    "first_name",
    "last_name",
    "email",
    "mobile",
    "phone",
    "designation",
    "department",
    "company_name",
)


@dataclass(frozen=True, slots=True)
class CustomerContext:
    customer_name: str
    orders: list[dict[str, Any]]
    quotations: list[dict[str, Any]]
    source_label: str
    company: dict[str, Any] | None = None
    contact: dict[str, Any] | None = None
    current_prices: dict[str, dict[str, Any]] | None = None


def parse_session_identity(session_id: str) -> tuple[str | None, str | None]:
    """Return (email, phone) derived from channel-bound session_id."""
    if ":" not in session_id:
        return None, None
    channel, raw_id = session_id.split(":", 1)
    raw_id = raw_id.strip()
    if not raw_id:
        return None, None
    if channel == "email":
        return raw_id.lower(), None
    if channel == "whatsapp":
        return None, raw_id
    return None, None


def session_display_label(session_id: str) -> str:
    """Human-readable session id for dashboard display (strips channel prefix)."""
    if ":" in session_id:
        return session_id.split(":", 1)[1]
    return session_id


def format_context(ctx: CustomerContext) -> str:
    lines = [f"Customer: {ctx.customer_name}", f"Source: {ctx.source_label}"]
    lines.extend(_format_profile_section("Company", ctx.company))
    lines.extend(_format_profile_section("Contact", ctx.contact, key_order=_CONTACT_FIELD_ORDER))
    lines.extend(_format_current_prices(ctx.current_prices))
    if ctx.orders:
        lines.append("Recent orders/invoices:")
        for row in ctx.orders:
            lines.append(
                _format_row(row, date_key="transaction_date", current_prices=ctx.current_prices)
            )
    if ctx.quotations:
        lines.append("Recent quotations/estimates:")
        for row in ctx.quotations:
            lines.append(
                _format_row(
                    row,
                    date_key="transaction_date",
                    extra="valid_till",
                    current_prices=ctx.current_prices,
                )
            )
    if len(lines) == 2:
        lines.append("No orders or quotations on file.")
    lines.append(
        "Use this data to answer the customer. Do not reveal internal system names or cite internal IDs unless asked."
    )
    return "\n".join(lines)


def _format_row(
    row: dict[str, Any],
    *,
    date_key: str,
    extra: str | None = None,
    current_prices: dict[str, dict[str, Any]] | None = None,
) -> str:
    name = row.get("name", "?")
    status = row.get("status", "?")
    date = row.get(date_key, "?")
    total = row.get("grand_total", "?")
    parts = [f"- {name} ({date}) status={status} total={total}"]
    if extra and row.get(extra):
        parts.append(f" valid_until={row[extra]}")
    line = "".join(parts)
    products = _format_products(row.get("items"), current_prices=current_prices)
    if products:
        line = f"{line}\n  products: {products}"
    return line


def _format_products(
    raw_items: Any,
    *,
    current_prices: dict[str, dict[str, Any]] | None = None,
) -> str:
    if not isinstance(raw_items, list) or not raw_items:
        return ""
    chunks: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("item_name") or item.get("item_code") or "?").strip()
        qty = item.get("qty", "?")
        chunk = f"{label} x{qty}"
        uom = item.get("uom")
        rate = item.get("rate")
        if uom:
            chunk = f"{chunk} {uom}"
        if rate is not None:
            chunk = f"{chunk} @{rate}"
        code = str(item.get("item_code") or "").strip()
        if current_prices and code and code in current_prices:
            current = current_prices[code].get("current_rate")
            if isinstance(current, (int, float)) and current > 0:
                chunk = f"{chunk}; current list @{current}"
        chunks.append(chunk)
    return "; ".join(chunks)


def _format_current_prices(prices: dict[str, dict[str, Any]] | None) -> list[str]:
    if not prices:
        return []
    lines: list[str] = []
    for code in sorted(prices):
        info = prices[code]
        rate = info.get("current_rate")
        if not isinstance(rate, (int, float)) or rate <= 0:
            continue
        uom = info.get("uom")
        currency = info.get("currency")
        chunk = f"- {code}: {rate}"
        if currency:
            chunk = f"{chunk} {currency}"
        if uom:
            chunk = f"{chunk}/{uom}"
        source = info.get("source")
        if source == "price_list" and info.get("price_list"):
            chunk = f"{chunk} ({info['price_list']})"
        lines.append(chunk)
    if not lines:
        return []
    return ["Current list prices (customer price list):"] + lines


def _format_profile_section(
    title: str,
    profile: dict[str, Any] | None,
    *,
    key_order: tuple[str, ...] | None = None,
) -> list[str]:
    if not profile:
        return []
    lines = [f"{title}:"]
    keys = list(key_order) if key_order else list(profile.keys())
    seen: set[str] = set()
    for key in keys:
        seen.add(key)
        value = profile.get(key)
        if key == "address" and isinstance(value, dict):
            formatted = _format_address(value)
            if formatted:
                lines.append(f"- address: {formatted}")
            continue
        if value is None or str(value).strip() == "":
            continue
        lines.append(f"- {key}: {value}")
    for key, value in profile.items():
        if key in seen or key == "address":
            continue
        if value is None or str(value).strip() == "":
            continue
        lines.append(f"- {key}: {value}")
    return lines


def _format_address(address: dict[str, Any]) -> str:
    parts = [
        str(address.get("line1") or "").strip(),
        str(address.get("line2") or "").strip(),
        str(address.get("city") or "").strip(),
        str(address.get("state") or "").strip(),
        str(address.get("pincode") or "").strip(),
        str(address.get("country") or "").strip(),
    ]
    return ", ".join(part for part in parts if part)


def _collect_item_codes(
    orders: list[dict[str, Any]], quotations: list[dict[str, Any]]
) -> list[str]:
    codes: list[str] = []
    for row in orders + quotations:
        items = row.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            code = str(item.get("item_code") or "").strip()
            if code:
                codes.append(code)
    return list(dict.fromkeys(codes))


class CustomerAccessGate:
    """Resolve customer context from channel identity only."""

    def __init__(
        self,
        client: CustomerDataClient,
        config: dict[str, Any],
        *,
        source_label: str,
        fetch_orders_key: str = "fetch_orders",
        fetch_quotations_key: str = "fetch_quotations",
    ) -> None:
        self._client = client
        self._source_label = source_label
        self._fetch_orders = _config_bool(config.get(fetch_orders_key), default=True)
        self._fetch_quotations = _config_bool(config.get(fetch_quotations_key), default=True)
        self._fetch_current_prices = _config_bool(config.get("fetch_current_prices"), default=True)
        self._max_items = max(1, _config_int(config.get("max_items"), default=5))

    def resolve(self, session_id: str) -> CustomerContext | None:
        email, phone = parse_session_identity(session_id)
        if not email and not phone:
            return None
        try:
            customer = self._client.find_customer(email=email, phone=phone)
        except Exception:
            logger.exception("%s customer lookup failed for session %s", self._source_label, session_id)
            return None
        if not customer:
            return None
        company: dict[str, Any] | None = None
        contact: dict[str, Any] | None = None
        orders: list[dict[str, Any]] = []
        quotations: list[dict[str, Any]] = []
        current_prices: dict[str, dict[str, Any]] | None = None
        try:
            company = self._client.get_customer_profile(customer)
            contact = self._client.get_matched_contact(
                email=email,
                phone=phone,
                customer=customer,
            )
            if self._fetch_orders:
                orders = self._client.get_orders(customer, self._max_items)
            if self._fetch_quotations:
                quotations = self._client.get_quotations(customer, self._max_items)
            if self._fetch_current_prices:
                codes = _collect_item_codes(orders, quotations)
                if codes:
                    current_prices = self._client.get_current_item_prices(customer, codes)
        except Exception:
            logger.exception("%s fetch failed for customer %s", self._source_label, customer)
        return CustomerContext(
            customer_name=customer,
            orders=orders,
            quotations=quotations,
            source_label=self._source_label,
            company=company or None,
            contact=contact,
            current_prices=current_prices or None,
        )

    def enrich(self, session_id: str) -> str | None:
        ctx = self.resolve(session_id)
        return format_context(ctx) if ctx else None


def resolve_manual_identity(*, test_email: str | None, test_phone: str | None) -> tuple[str | None, str | None]:
    email = test_email.strip().lower() if test_email and test_email.strip() else None
    phone = test_phone.strip() if test_phone and test_phone.strip() else None
    return email, phone


def _config_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _config_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
