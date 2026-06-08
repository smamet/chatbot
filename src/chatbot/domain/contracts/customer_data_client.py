from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CustomerDataClient(Protocol):
    def find_customer(self, *, email: str | None = None, phone: str | None = None) -> str | None:
        """Return a stable customer identifier for downstream queries."""

    def get_orders(self, customer: str, limit: int) -> list[dict[str, Any]]:
        """Return recent orders/invoices for the customer."""

    def get_quotations(self, customer: str, limit: int) -> list[dict[str, Any]]:
        """Return recent quotations/estimates for the customer."""

    def get_customer_profile(self, customer: str) -> dict[str, Any]:
        """Return company/customer master data for prompt enrichment."""

    def get_matched_contact(
        self,
        *,
        email: str | None = None,
        phone: str | None = None,
        customer: str,
    ) -> dict[str, Any] | None:
        """Return the matched contact person when channel identity is known."""

    def ping(self) -> None:
        """Verify credentials and API reachability."""
