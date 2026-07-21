from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.quickbooks.client import QuickBooksClient
from chatbot.application.customer_access_gate import CustomerAccessGate
from chatbot.application.integration_service import IntegrationService
from chatbot.domain.models.integration import IntegrationType

_PRIORITY = (IntegrationType.ERPNEXT, IntegrationType.QUICKBOOKS)


def build_enricher(session: Session, tenant_id: int) -> Callable[[str], str | None] | None:
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    gates: list[CustomerAccessGate] = []
    for itype in _PRIORITY:
        integration = svc.find_active(tenant_id, type=itype)
        if integration is None:
            continue
        gate = _gate_for_integration(integration.type, integration.config)
        if gate is not None:
            gates.append(gate)
    if not gates:
        return None

    def enrich(session_id: str) -> str | None:
        for gate in gates:
            block = gate.enrich(session_id)
            if block:
                return block
        return None

    return enrich


def _gate_for_integration(
    integration_type: IntegrationType,
    config: dict,
) -> CustomerAccessGate | None:
    if integration_type == IntegrationType.ERPNEXT:
        return CustomerAccessGate(
            ErpNextClient(config),
            config,
            source_label="ERPNext",
            fetch_orders_key="fetch_orders",
            fetch_quotations_key="fetch_quotations",
        )
    if integration_type == IntegrationType.QUICKBOOKS:
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
