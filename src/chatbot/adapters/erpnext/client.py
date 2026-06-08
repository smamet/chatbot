from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote, urljoin

import httpx

logger = logging.getLogger(__name__)

_ORDER_FIELDS = ["name", "transaction_date", "status", "grand_total", "delivery_date"]
_INVOICE_FIELDS = ["name", "posting_date", "status", "grand_total"]
_QUOTATION_FIELDS = ["name", "transaction_date", "status", "grand_total", "valid_till"]
_ITEM_FIELDS = ["item_code", "item_name", "standard_rate", "stock_uom"]
_LINE_ITEM_FIELDS = ["item_code", "item_name", "qty", "rate", "amount", "uom"]


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

    def _post(self, path: str, *, json_body: dict[str, Any]) -> dict[str, Any]:
        if not self._base_url or not self._api_key or not self._api_secret:
            return {}
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=self._headers(), json=json_body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("ERPNext POST failed %s: %s", path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def _get_bytes(self, path: str, *, params: dict[str, str] | None = None) -> bytes:
        if not self._base_url or not self._api_key or not self._api_secret:
            return b""
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, headers=self._headers(), params=params or {})
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as exc:
            logger.warning("ERPNext binary GET failed %s: %s", path, exc)
            return b""

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

    def _get_resource(self, doctype: str, name: str) -> dict[str, Any]:
        safe_name = name.strip()
        if not safe_name:
            return {}
        encoded = quote(safe_name, safe="")
        data = self._get(f"/api/resource/{doctype}/{encoded}")
        doc = data.get("data")
        return doc if isinstance(doc, dict) else {}

    def _find_contact(
        self, *, email: str | None = None, phone: str | None = None
    ) -> dict[str, Any] | None:
        contact_name = self._find_contact_name(email=email, phone=phone)
        if not contact_name:
            return None
        # List API omits child tables; fetch full Contact doc for `links`.
        contact = self._get_resource("Contact", contact_name)
        return contact or None

    def _find_contact_name(
        self, *, email: str | None = None, phone: str | None = None
    ) -> str | None:
        if email:
            rows = self._list_resource(
                "Contact",
                filters=[[self._email_field, "=", email]],
                fields=["name"],
                limit=1,
            )
            if rows:
                name = str(rows[0].get("name", "")).strip()
                if name:
                    return name
        if phone:
            for candidate in _phone_variants(phone):
                rows = self._list_resource(
                    "Contact",
                    filters=[[self._phone_field, "=", candidate]],
                    fields=["name"],
                    limit=1,
                )
                if rows:
                    name = str(rows[0].get("name", "")).strip()
                    if name:
                        return name
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

    def get_sales_invoices(self, customer: str, limit: int) -> list[dict[str, Any]]:
        rows = self._list_resource(
            "Sales Invoice",
            filters=[["customer", "=", customer]],
            fields=_INVOICE_FIELDS,
            limit=limit,
            order_by="posting_date desc",
        )
        return self._attach_line_items("Sales Invoice", rows)

    def get_quotations(self, customer: str, limit: int) -> list[dict[str, Any]]:
        rows = self._list_resource(
            "Quotation",
            filters=[["party_name", "=", customer], ["quotation_to", "=", "Customer"]],
            fields=_QUOTATION_FIELDS,
            limit=limit,
            order_by="transaction_date desc",
        )
        return self._attach_line_items("Quotation", rows)

    def get_orders(self, customer: str, limit: int) -> list[dict[str, Any]]:
        return self.get_sales_invoices(customer, limit)

    def get_customer_profile(self, customer: str) -> dict[str, Any]:
        doc = self._get_resource("Customer", customer)
        if not doc:
            return {"name": customer}
        profile = _normalize_customer_profile(doc)
        profile["address"] = self._get_customer_address(customer, doc)
        return profile

    def get_matched_contact(
        self,
        *,
        email: str | None = None,
        phone: str | None = None,
        customer: str,
    ) -> dict[str, Any] | None:
        contact_doc = self._find_contact(email=email, phone=phone)
        if contact_doc:
            return _normalize_contact(contact_doc)
        customer_doc = self._get_resource("Customer", customer)
        primary = str(customer_doc.get("customer_primary_contact") or "").strip()
        if not primary:
            return None
        return _normalize_contact(self._get_resource("Contact", primary))

    def _get_customer_address(
        self, customer: str, customer_doc: dict[str, Any]
    ) -> dict[str, Any] | None:
        primary = str(customer_doc.get("customer_primary_address") or "").strip()
        if primary:
            return _normalize_address(self._get_resource("Address", primary))
        rows = self._list_resource(
            "Address",
            filters=[
                ["Dynamic Link", "link_name", "=", customer],
                ["Dynamic Link", "link_doctype", "=", "Customer"],
            ],
            fields=["name"],
            limit=1,
        )
        if not rows:
            return None
        name = str(rows[0].get("name", "")).strip()
        if not name:
            return None
        return _normalize_address(self._get_resource("Address", name))

    def _attach_line_items(
        self, doctype: str, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for row in rows:
            name = str(row.get("name", "")).strip()
            base = _normalize_document_row(row)
            if not name:
                enriched.append(base)
                continue
            doc = self._get_resource(doctype, name)
            base["items"] = _normalize_line_items(doc.get("items"))
            enriched.append(base)
        return enriched

    def ping(self) -> None:
        self._list_resource("Customer", filters=[], fields=["name"], limit=1)

    def search_items(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        token = query.strip()
        if not token:
            return []
        rows = self._list_resource(
            "Item",
            filters=[["item_name", "like", f"%{token}%"]],
            fields=_ITEM_FIELDS,
            limit=limit,
        )
        if rows:
            return rows
        return self._list_resource(
            "Item",
            filters=[["item_code", "like", f"%{token}%"]],
            fields=_ITEM_FIELDS,
            limit=limit,
        )

    def get_item_by_code(self, item_code: str) -> dict[str, Any] | None:
        code = item_code.strip()
        if not code:
            return None
        rows = self._list_resource(
            "Item",
            filters=[["item_code", "=", code]],
            fields=_ITEM_FIELDS,
            limit=1,
        )
        return rows[0] if rows else None

    def create_quotation(
        self,
        customer: str,
        lines: list[dict[str, Any]],
        *,
        notes: str | None = None,
    ) -> dict[str, Any]:
        items = [
            {
                "item_code": line["item_code"],
                "qty": line["qty"],
                "rate": line.get("rate"),
            }
            for line in lines
        ]
        body: dict[str, Any] = {
            "quotation_to": "Customer",
            "party_name": customer,
            "items": items,
        }
        if notes:
            body["terms"] = notes
        result = self._post("/api/resource/Quotation", json_body={"data": body})
        data = result.get("data")
        return data if isinstance(data, dict) else {}

    def download_quotation_pdf(self, name: str) -> bytes:
        safe = name.strip()
        if not safe:
            return b""
        return self._get_bytes(
            "/api/method/frappe.utils.print_format.download_pdf",
            params={
                "doctype": "Quotation",
                "name": safe,
                "format": "Standard",
            },
        )


def _normalize_customer_profile(doc: dict[str, Any]) -> dict[str, Any]:
    name = str(doc.get("customer_name") or doc.get("name") or "").strip()
    return _compact_profile(
        {
            "name": name,
            "customer_type": doc.get("customer_type"),
            "customer_group": doc.get("customer_group"),
            "territory": doc.get("territory"),
            "tax_id": doc.get("tax_id"),
            "website": doc.get("website"),
            "email": doc.get("email_id"),
            "mobile": doc.get("mobile_no"),
            "phone": doc.get("phone"),
        }
    )


def _normalize_contact(doc: dict[str, Any]) -> dict[str, Any]:
    if not doc:
        return {}
    first = str(doc.get("first_name") or "").strip()
    last = str(doc.get("last_name") or "").strip()
    full_name = " ".join(part for part in (first, last) if part).strip()
    return _compact_profile(
        {
            "name": full_name or str(doc.get("name") or "").strip(),
            "first_name": first or None,
            "last_name": last or None,
            "email": doc.get("email_id"),
            "mobile": doc.get("mobile_no"),
            "phone": doc.get("phone"),
            "designation": doc.get("designation"),
            "department": doc.get("department"),
            "company_name": doc.get("company_name"),
        }
    )


def _normalize_address(doc: dict[str, Any]) -> dict[str, Any] | None:
    if not doc:
        return None
    return _compact_profile(
        {
            "title": doc.get("address_title"),
            "line1": doc.get("address_line1"),
            "line2": doc.get("address_line2"),
            "city": doc.get("city"),
            "state": doc.get("state"),
            "country": doc.get("country"),
            "pincode": doc.get("pincode"),
            "phone": doc.get("phone"),
            "email": doc.get("email_id"),
        }
    )


def _compact_profile(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if value is not None and str(value).strip() != ""
    }


def _normalize_document_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if not out.get("transaction_date"):
        posting = out.get("posting_date")
        if posting:
            out["transaction_date"] = posting
    return out


def _normalize_line_items(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, Any]] = []
    for row in raw_items:
        if not isinstance(row, dict):
            continue
        code = str(row.get("item_code", "")).strip()
        name = str(row.get("item_name", "")).strip()
        if not code and not name:
            continue
        items.append(
            {
                "item_code": code or name,
                "item_name": name or code,
                "qty": row.get("qty"),
                "rate": row.get("rate"),
                "amount": row.get("amount"),
                "uom": row.get("uom"),
            }
        )
    return items


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
