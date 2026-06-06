from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from chatbot.domain.contracts.customer_data_client import CustomerDataClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CustomerContext:
    customer_name: str
    orders: list[dict[str, Any]]
    quotations: list[dict[str, Any]]
    source_label: str


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


def format_context(ctx: CustomerContext) -> str:
    lines = [f"Customer: {ctx.customer_name}", f"Source: {ctx.source_label}"]
    if ctx.orders:
        lines.append("Recent orders/invoices:")
        for row in ctx.orders:
            lines.append(_format_row(row, date_key="transaction_date"))
    if ctx.quotations:
        lines.append("Recent quotations/estimates:")
        for row in ctx.quotations:
            lines.append(_format_row(row, date_key="transaction_date", extra="valid_till"))
    if len(lines) == 2:
        lines.append("No orders or quotations on file.")
    lines.append(
        "Use this data to answer the customer. Do not reveal internal system names or cite internal IDs unless asked."
    )
    return "\n".join(lines)


def _format_row(row: dict[str, Any], *, date_key: str, extra: str | None = None) -> str:
    name = row.get("name", "?")
    status = row.get("status", "?")
    date = row.get(date_key, "?")
    total = row.get("grand_total", "?")
    parts = [f"- {name} ({date}) status={status} total={total}"]
    if extra and row.get(extra):
        parts.append(f" valid_until={row[extra]}")
    return "".join(parts)


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
        orders: list[dict[str, Any]] = []
        quotations: list[dict[str, Any]] = []
        try:
            if self._fetch_orders:
                orders = self._client.get_orders(customer, self._max_items)
            if self._fetch_quotations:
                quotations = self._client.get_quotations(customer, self._max_items)
        except Exception:
            logger.exception("%s fetch failed for customer %s", self._source_label, customer)
        return CustomerContext(
            customer_name=customer,
            orders=orders,
            quotations=quotations,
            source_label=self._source_label,
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
