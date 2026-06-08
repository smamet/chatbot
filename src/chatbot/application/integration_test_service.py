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
    session_id = f"email:{email}" if email else f"whatsapp:{phone}"
    try:
        ctx = gate.resolve(session_id)
    except Exception as exc:
        return IntegrationTestResult(
            ok=False,
            message="Customer lookup failed.",
            error=str(exc),
        )
    if ctx is None:
        return IntegrationTestResult(
            ok=True,
            message="Connection OK, but no customer matched the test identity.",
            customer=None,
            orders=[],
            quotations=[],
            preview="Connection OK — no matching customer.",
        )
    preview_limit = min(3, gate._max_items)  # noqa: SLF001
    preview = format_context(
        CustomerContext(
            customer_name=ctx.customer_name,
            orders=ctx.orders[:preview_limit],
            quotations=ctx.quotations[:preview_limit],
            source_label=ctx.source_label,
            company=ctx.company,
            contact=ctx.contact,
            current_prices=ctx.current_prices,
        )
    )
    return IntegrationTestResult(
        ok=True,
        message=f"Connection OK — customer {ctx.customer_name} found.",
        customer=ctx.customer_name,
        orders=ctx.orders[:preview_limit],
        quotations=ctx.quotations[:preview_limit],
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
