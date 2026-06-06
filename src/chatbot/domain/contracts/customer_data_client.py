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

    def ping(self) -> None:
        """Verify credentials and API reachability."""
