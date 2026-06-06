from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

_ORDER_FIELDS = ["name", "transaction_date", "status", "grand_total", "delivery_date"]
_QUOTATION_FIELDS = ["name", "transaction_date", "status", "grand_total", "valid_till"]


class ErpNextClient:
    """Thin REST client for ERPNext (Frappe) resource API."""

    def __init__(self, config: dict[str, Any], *, timeout: float = 15.0) -> None:
        self._base_url = str(config.get("url", "")).strip().rstrip("/")
        self._api_key = str(config.get("api_key", "")).strip()
        self._api_secret = str(config.get("api_secret", "")).strip()
        self._email_field = str(config.get("identity_email_field", "email_id")).strip() or "email_id"
        self._phone_field = str(config.get("identity_phone_field", "mobile_no")).strip() or "mobile_no"
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self._api_key}:{self._api_secret}"}

    def _get(self, path: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
        if not self._base_url or not self._api_key or not self._api_secret:
            return {}
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, headers=self._headers(), params=params or {})
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("ERPNext request failed %s: %s", path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _filters_json(filters: list[list[str]]) -> str:
        return json.dumps(filters)

    def _list_resource(
        self,
        doctype: str,
        *,
        filters: list[list[str]],
        fields: list[str],
        limit: int,
        order_by: str = "modified desc",
    ) -> list[dict[str, Any]]:
        params = {
            "filters": self._filters_json(filters),
            "fields": json.dumps(fields),
            "limit_page_length": str(max(1, limit)),
            "order_by": order_by,
        }
        data = self._get(f"/api/resource/{doctype}", params=params)
        rows = data.get("data")
        return rows if isinstance(rows, list) else []

    def find_customer(self, *, email: str | None = None, phone: str | None = None) -> str | None:
        contact = self._find_contact(email=email, phone=phone)
        if not contact:
            return None
        return self._customer_from_contact(contact)

    def _find_contact(
        self, *, email: str | None = None, phone: str | None = None
    ) -> dict[str, Any] | None:
        if email:
            rows = self._list_resource(
                "Contact",
                filters=[[self._email_field, "=", email]],
                fields=["name", "links"],
                limit=1,
            )
            if rows:
                return rows[0]
        if phone:
            for candidate in _phone_variants(phone):
                rows = self._list_resource(
                    "Contact",
                    filters=[[self._phone_field, "=", candidate]],
                    fields=["name", "links"],
                    limit=1,
                )
                if rows:
                    return rows[0]
        return None

    def _customer_from_contact(self, contact: dict[str, Any]) -> str | None:
        links = contact.get("links")
        if not isinstance(links, list):
            return None
        for link in links:
            if not isinstance(link, dict):
                continue
            if link.get("link_doctype") == "Customer" and link.get("link_name"):
                return str(link["link_name"])
        return None

    def get_sales_orders(self, customer: str, limit: int) -> list[dict[str, Any]]:
        return self._list_resource(
            "Sales Order",
            filters=[["customer", "=", customer]],
            fields=_ORDER_FIELDS,
            limit=limit,
            order_by="transaction_date desc",
        )

    def get_quotations(self, customer: str, limit: int) -> list[dict[str, Any]]:
        return self._list_resource(
            "Quotation",
            filters=[["party_name", "=", customer], ["quotation_to", "=", "Customer"]],
            fields=_QUOTATION_FIELDS,
            limit=limit,
            order_by="transaction_date desc",
        )

    def get_orders(self, customer: str, limit: int) -> list[dict[str, Any]]:
        return self.get_sales_orders(customer, limit)

    def ping(self) -> None:
        self._list_resource("Customer", filters=[], fields=["name"], limit=1)


def _phone_variants(phone: str) -> list[str]:
    raw = phone.strip()
    if not raw:
        return []
    digits = "".join(ch for ch in raw if ch.isdigit())
    variants = [raw]
    if digits and digits not in variants:
        variants.append(digits)
    if digits.startswith("00") and len(digits) > 2:
        plus = f"+{digits[2:]}"
        if plus not in variants:
            variants.append(plus)
    if not raw.startswith("+") and digits:
        plus = f"+{digits}"
        if plus not in variants:
            variants.append(plus)
    return variants
