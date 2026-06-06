from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.quickbooks.client import QuickBooksClient
from chatbot.application.customer_access_gate import (
    CustomerAccessGate,
    CustomerContext,
    format_context,
    resolve_manual_identity,
)
from chatbot.domain.models.integration import IntegrationType


@dataclass(frozen=True, slots=True)
class IntegrationTestResult:
    ok: bool
    message: str
    customer: str | None = None
    orders: list[dict[str, Any]] | None = None
    quotations: list[dict[str, Any]] | None = None
    preview: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_integration_test(
    integration_type: str,
    config: dict[str, Any],
    *,
    test_email: str | None = None,
    test_phone: str | None = None,
) -> IntegrationTestResult:
    email, phone = resolve_manual_identity(test_email=test_email or "", test_phone=test_phone or "")
    if not email and not phone:
        return IntegrationTestResult(
            ok=False,
            message="Provide a test email and/or phone number.",
            error="missing_test_identity",
        )
    gate = _gate_for_type(integration_type, config)
    if gate is None:
        return IntegrationTestResult(
            ok=False,
            message="Unsupported or incomplete integration configuration.",
            error="invalid_integration",
        )
    try:
        gate._client.ping()  # noqa: SLF001
    except Exception as exc:
        return IntegrationTestResult(
            ok=False,
            message="Connection failed.",
            error=str(exc),
        )
    try:
        customer = gate._client.find_customer(email=email, phone=phone)  # noqa: SLF001
    except Exception as exc:
        return IntegrationTestResult(
            ok=False,
            message="Customer lookup failed.",
            error=str(exc),
        )
    if not customer:
        return IntegrationTestResult(
            ok=True,
            message="Connection OK, but no customer matched the test identity.",
            customer=None,
            orders=[],
            quotations=[],
            preview="Connection OK — no matching customer.",
        )
    orders: list[dict[str, Any]] = []
    quotations: list[dict[str, Any]] = []
    try:
        preview_limit = min(3, gate._max_items)  # noqa: SLF001
        if gate._fetch_orders:  # noqa: SLF001
            orders = gate._client.get_orders(customer, preview_limit)  # noqa: SLF001
        if gate._fetch_quotations:  # noqa: SLF001
            quotations = gate._client.get_quotations(customer, preview_limit)  # noqa: SLF001
    except Exception as exc:
        return IntegrationTestResult(
            ok=False,
            message=f"Customer found ({customer}), but fetching records failed.",
            customer=customer,
            error=str(exc),
        )
    preview = format_context(
        CustomerContext(
            customer_name=customer,
            orders=orders,
            quotations=quotations,
            source_label=gate._source_label,  # noqa: SLF001
        )
    )
    return IntegrationTestResult(
        ok=True,
        message=f"Connection OK — customer {customer} found.",
        customer=customer,
        orders=orders,
        quotations=quotations,
        preview=preview,
    )


def _gate_for_type(integration_type: str, config: dict[str, Any]) -> CustomerAccessGate | None:
    try:
        itype = IntegrationType(integration_type)
    except ValueError:
        return None
    if itype == IntegrationType.ERPNEXT:
        if not str(config.get("url", "")).strip():
            return None
        return CustomerAccessGate(
            ErpNextClient(config),
            config,
            source_label="ERPNext",
            fetch_orders_key="fetch_orders",
            fetch_quotations_key="fetch_quotations",
        )
    if itype == IntegrationType.QUICKBOOKS:
        if not str(config.get("refresh_token", "")).strip():
            return None
        return CustomerAccessGate(
            QuickBooksClient(config),
            config,
            source_label="QuickBooks",
            fetch_orders_key="fetch_invoices",
            fetch_quotations_key="fetch_estimates",
        )
    return None
