from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from chatbot.application.progress_log import ProgressLog

logger = logging.getLogger(__name__)

_ORDER_FIELDS = ["name", "transaction_date", "status", "grand_total", "delivery_date"]
_INVOICE_FIELDS = ["name", "posting_date", "status", "grand_total"]
_QUOTATION_FIELDS = ["name", "transaction_date", "status", "grand_total", "valid_till"]
_ITEM_FIELDS = ["item_code", "item_name", "standard_rate", "stock_uom"]
_ITEM_PRICE_FIELDS = ["item_code", "price_list", "price_list_rate", "currency", "uom"]
_LINE_ITEM_FIELDS = ["item_code", "item_name", "qty", "rate", "amount", "uom"]


class ErpNextClient:
    """Thin REST client for ERPNext (Frappe) resource API."""

    def __init__(self, config: dict[str, Any], *, timeout: float = 15.0) -> None:
        self._base_url = str(config.get("url", "")).strip().rstrip("/")
        self._api_key = str(config.get("api_key", "")).strip()
        self._api_secret = str(config.get("api_secret", "")).strip()
        self._email_field = str(config.get("identity_email_field", "email_id")).strip() or "email_id"
        self._phone_field = str(config.get("identity_phone_field", "mobile_no")).strip() or "mobile_no"
        self._quotation_print_format = str(config.get("quotation_print_format", "")).strip() or None
        self._timeout = timeout
        self._cached_contact: dict[str, Any] | None = None
        self.last_pdf_error: str | None = None

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

    @staticmethod
    def _parse_erpnext_error(exc: httpx.HTTPStatusError) -> str:
        try:
            payload = exc.response.json()
        except (json.JSONDecodeError, ValueError):
            return (exc.response.text or str(exc))[:500]
        if not isinstance(payload, dict):
            return str(exc)
        if payload.get("message"):
            return str(payload["message"])
        server_messages = payload.get("_server_messages")
        if server_messages:
            try:
                raw = json.loads(server_messages) if isinstance(server_messages, str) else server_messages
                parts: list[str] = []
                for item in raw if isinstance(raw, list) else []:
                    if isinstance(item, str):
                        item = json.loads(item)
                    if isinstance(item, dict) and item.get("message"):
                        parts.append(str(item["message"]))
                if parts:
                    return "; ".join(parts)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if payload.get("exception"):
            return str(payload["exception"]).strip().split("\n")[-1]
        return str(exc)

    def _post_write(self, path: str, *, json_body: dict[str, Any]) -> dict[str, Any]:
        if not self._base_url or not self._api_key or not self._api_secret:
            return {"_erpnext_error": "ERPNext credentials are not configured"}
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=self._headers(), json=json_body)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_erpnext_error(exc)
            logger.warning("ERPNext POST failed %s: %s", path, detail)
            return {"_erpnext_error": detail}
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("ERPNext POST failed %s: %s", path, exc)
            return {"_erpnext_error": str(exc)}
        return payload if isinstance(payload, dict) else {"_erpnext_error": "Invalid ERPNext response"}

    def _put(self, path: str, *, json_body: dict[str, Any]) -> dict[str, Any]:
        if not self._base_url or not self._api_key or not self._api_secret:
            return {}
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.put(url, headers=self._headers(), json=json_body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("ERPNext PUT failed %s: %s", path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def _get_bytes(self, path: str, *, params: dict[str, str] | None = None) -> bytes:
        data, _err = self._fetch_bytes(path, params=params)
        return data

    def _fetch_bytes(self, path: str, *, params: dict[str, str] | None = None) -> tuple[bytes, str | None]:
        if not self._base_url or not self._api_key or not self._api_secret:
            return b"", "ERPNext credentials are not configured"
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, headers=self._headers(), params=params or {})
                response.raise_for_status()
                return self._parse_pdf_response(response.content)
        except httpx.HTTPStatusError as exc:
            return b"", self._parse_erpnext_error(exc)
        except httpx.HTTPError as exc:
            return b"", str(exc)

    @staticmethod
    def _parse_pdf_response(data: bytes) -> tuple[bytes, str | None]:
        if data.startswith(b"%PDF"):
            return data, None
        if not data:
            return b"", "Empty response"
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            snippet = data[:200].decode("utf-8", errors="replace")
            return b"", f"Unexpected response: {snippet}"
        if not isinstance(payload, dict):
            return b"", "Invalid ERPNext PDF response"
        err = ErpNextClient._pdf_error_from_payload(payload)
        if err:
            return b"", err
        return b"", "ERPNext returned JSON instead of PDF"

    @staticmethod
    def _pdf_error_from_payload(payload: dict[str, Any]) -> str | None:
        if payload.get("exc"):
            exc = str(payload["exc"]).strip()
            return exc.split("\n")[-1][:500]
        if payload.get("message"):
            return str(payload["message"])[:500]
        server_messages = payload.get("_server_messages")
        if server_messages:
            try:
                raw = json.loads(server_messages) if isinstance(server_messages, str) else server_messages
                parts: list[str] = []
                for item in raw if isinstance(raw, list) else []:
                    if isinstance(item, str):
                        item = json.loads(item)
                    if isinstance(item, dict) and item.get("message"):
                        parts.append(str(item["message"]))
                if parts:
                    return "; ".join(parts)[:500]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _pdf_failure_hint(error: str) -> str:
        lowered = error.lower()
        if "broken image links" in lowered:
            return (
                " ERPNext could not load images while rendering the PDF (letterhead/logo URLs). "
                "Check site_config.json host_name and that the ERPNext server can reach its own /files URLs. "
                "If PDF works in the ERPNext UI but not via API, try Print Designer / Chromium PDF or fix image paths."
            )
        if "not found" in lowered and "print format" in lowered:
            return " Set the correct format name under Integrations → Quotation print format."
        return ""

    def _default_quotation_print_format(self) -> str | None:
        rows = self._list_resource(
            "Property Setter",
            filters=[
                ["doc_type", "=", "Quotation"],
                ["property", "=", "default_print_format"],
            ],
            fields=["value"],
            limit=1,
        )
        if rows:
            value = str(rows[0].get("value", "")).strip()
            if value:
                return value
        rows = self._list_resource(
            "Print Format",
            filters=[["doc_type", "=", "Quotation"], ["standard", "=", "Yes"]],
            fields=["name"],
            limit=1,
        )
        if rows:
            value = str(rows[0].get("name", "")).strip()
            if value:
                return value
        return None

    def _quotation_pdf_format_candidates(self) -> list[str | None]:
        discovered = self._discover_quotation_print_formats()
        discovered_set = set(discovered)
        formats: list[str | None] = []
        seen: set[str | None] = set()

        def _add(fmt: str | None) -> None:
            if fmt not in seen:
                seen.add(fmt)
                formats.append(fmt)

        _add(self._quotation_print_format)
        _add(self._default_quotation_print_format())
        _add(None)
        for fmt in discovered:
            _add(fmt)
        if "Standard" in discovered_set or not discovered_set:
            _add("Standard")
        return formats

    def download_quotation_pdf(self, name: str, *, on_log: ProgressLog | None = None) -> bytes:
        log = on_log.step if on_log is not None else None
        self.last_pdf_error = None
        safe = name.strip()
        if not safe:
            self.last_pdf_error = "Missing quotation name"
            return b""

        formats = self._quotation_pdf_format_candidates()
        default_fmt = self._default_quotation_print_format()
        discovered = self._discover_quotation_print_formats()
        if log is not None:
            if default_fmt:
                log(f"ERPNext default print format: {default_fmt}")
            if discovered:
                log(f"Available print formats: {', '.join(discovered)}")

        endpoints = (
            "/api/method/frappe.templates.pages.print.download_pdf",
            "/api/method/frappe.utils.print_format.download_pdf",
        )
        errors: list[str] = []
        broken_image_errors = 0
        for endpoint in endpoints:
            endpoint_short = endpoint.rsplit("/", 1)[-1]
            for print_format in formats:
                fmt_label = print_format or default_fmt or "(ERPNext default)"
                for no_letterhead in ("0", "1"):
                    letterhead_label = "letterhead off" if no_letterhead == "1" else "letterhead on"
                    params: dict[str, str] = {
                        "doctype": "Quotation",
                        "name": safe,
                        "no_letterhead": no_letterhead,
                    }
                    if print_format:
                        params["format"] = print_format
                    if log is not None:
                        log(f"PDF attempt: {fmt_label}, {letterhead_label} ({endpoint_short})…")
                    data, err = self._fetch_bytes(endpoint, params=params)
                    if data.startswith(b"%PDF"):
                        if log is not None:
                            log("PDF received from ERPNext")
                        return data
                    if err:
                        fmt_key = print_format or "(default)"
                        errors.append(f"{fmt_label}: {err}")
                        if "broken image links" in err.lower():
                            broken_image_errors += 1
                        if log is not None:
                            log(f"  Failed: {err}")
                        logger.debug(
                            "ERPNext PDF failed endpoint=%s format=%s: %s",
                            endpoint,
                            fmt_key,
                            err,
                        )
            if broken_image_errors and errors:
                break

        unique_errors = list(dict.fromkeys(errors))
        primary = unique_errors[0] if unique_errors else "No PDF returned from ERPNext"
        hint = self._pdf_failure_hint(primary)
        if len(unique_errors) > 1:
            self.last_pdf_error = f"{primary} (also tried {len(unique_errors) - 1} other format(s))"
        else:
            self.last_pdf_error = primary
        if hint:
            self.last_pdf_error += hint
        logger.warning("ERPNext quotation PDF failed for %s: %s", safe, self.last_pdf_error)
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
        limit_start: int = 0,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "filters": self._filters_json(filters),
            "fields": json.dumps(fields),
            "limit_page_length": str(max(1, limit)),
            "order_by": order_by,
        }
        if limit_start > 0:
            params["limit_start"] = str(limit_start)
        data = self._get(f"/api/resource/{doctype}", params=params)
        rows = data.get("data")
        return rows if isinstance(rows, list) else []

    def _list_resource_paginated(
        self,
        doctype: str,
        *,
        filters: list[list[str]],
        fields: list[str],
        page_length: int = 500,
        order_by: str = "modified asc",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        start = 0
        page_size = max(1, page_length)
        while True:
            page = self._list_resource(
                doctype,
                filters=filters,
                fields=fields,
                limit=page_size,
                order_by=order_by,
                limit_start=start,
            )
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size
        return rows

    _CATALOG_ITEM_FIELDS = [
        "item_code",
        "item_name",
        "description",
        "standard_rate",
        "stock_uom",
        "item_group",
        "modified",
        "disabled",
    ]

    def list_catalog_items(self, *, page_length: int = 500) -> list[dict[str, Any]]:
        rows = self._list_resource_paginated(
            "Item",
            filters=[["disabled", "=", 0]],
            fields=self._CATALOG_ITEM_FIELDS,
            page_length=page_length,
        )
        return [row for row in rows if str(row.get("item_code", "")).strip()]

    def fetch_stock_totals(self, *, page_length: int = 500) -> dict[str, float]:
        rows = self._list_resource_paginated(
            "Bin",
            filters=[],
            fields=["item_code", "actual_qty"],
            page_length=page_length,
        )
        totals: dict[str, float] = {}
        for row in rows:
            code = str(row.get("item_code", "")).strip()
            if not code:
                continue
            try:
                qty = float(row.get("actual_qty") or 0)
            except (TypeError, ValueError):
                qty = 0.0
            totals[code] = totals.get(code, 0.0) + qty
        return totals

    def fetch_price_list_rates(
        self,
        price_list: str,
        *,
        page_length: int = 500,
    ) -> dict[str, dict[str, Any]]:
        name = str(price_list or "").strip()
        if not name:
            return {}
        rows = self._list_resource_paginated(
            "Item Price",
            filters=[["price_list", "=", name]],
            fields=_ITEM_PRICE_FIELDS,
            page_length=page_length,
        )
        prices: dict[str, dict[str, Any]] = {}
        for row in rows:
            code = str(row.get("item_code", "")).strip()
            rate = _positive_rate(row.get("price_list_rate"))
            if not code or rate is None:
                continue
            prices[code] = {
                "rate": rate,
                "currency": row.get("currency"),
                "uom": row.get("uom"),
                "price_list": name,
            }
        return prices

    def find_customer(self, *, email: str | None = None, phone: str | None = None) -> str | None:
        contact = self._find_contact(email=email, phone=phone)
        if not contact:
            self._cached_contact = None
            return None
        self._cached_contact = contact
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
        self,
        *,
        email: str | None = None,
        phone: str | None = None,
        customer: str | None = None,
    ) -> dict[str, Any] | None:
        names = self._find_contact_names(email=email, phone=phone)
        if not names:
            return None
        docs = [self._get_resource("Contact", name) for name in names]
        docs = [doc for doc in docs if doc]
        if not docs:
            return None
        return _pick_best_contact(docs, customer)

    def _find_contact_names(
        self, *, email: str | None = None, phone: str | None = None
    ) -> list[str]:
        names: list[str] = []
        if email:
            names.extend(self._find_contact_names_by_email(email))
        if phone:
            names.extend(self._find_contact_names_by_phone(phone))
        return _dedupe_preserve_order(names)

    def _find_contact_names_by_email(self, email: str) -> list[str]:
        names: list[str] = []
        rows = self._list_resource(
            "Contact",
            filters=[[self._email_field, "=", email]],
            fields=["name"],
            limit=20,
        )
        names.extend(_contact_names_from_rows(rows))
        rows = self._list_resource(
            "Contact",
            filters=[["Contact Email", "email_id", "=", email]],
            fields=["name"],
            limit=20,
        )
        names.extend(_contact_names_from_rows(rows))
        return names

    def _find_contact_names_by_phone(self, phone: str) -> list[str]:
        names: list[str] = []
        for candidate in _phone_variants(phone):
            rows = self._list_resource(
                "Contact",
                filters=[[self._phone_field, "=", candidate]],
                fields=["name"],
                limit=20,
            )
            names.extend(_contact_names_from_rows(rows))
        return names

    def _find_contact_name(
        self, *, email: str | None = None, phone: str | None = None
    ) -> str | None:
        names = self._find_contact_names(email=email, phone=phone)
        return names[0] if names else None

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
        if self._cached_contact:
            return _normalize_contact(self._cached_contact)
        contact_doc = self._find_contact(email=email, phone=phone, customer=customer)
        if contact_doc:
            self._cached_contact = contact_doc
            return _normalize_contact(contact_doc)
        customer_doc = self._get_resource("Customer", customer)
        primary = str(customer_doc.get("customer_primary_contact") or "").strip()
        if not primary:
            return None
        contact_doc = self._get_resource("Contact", primary)
        if contact_doc:
            self._cached_contact = contact_doc
        return _normalize_contact(contact_doc)

    def get_current_item_prices(
        self, customer: str, item_codes: list[str]
    ) -> dict[str, dict[str, Any]]:
        codes = _dedupe_preserve_order(
            [str(code).strip() for code in item_codes if str(code).strip()]
        )
        if not codes:
            return {}
        customer_doc = self._get_resource("Customer", customer)
        price_list = str(customer_doc.get("default_price_list") or "").strip()
        prices: dict[str, dict[str, Any]] = {}
        if price_list:
            rows = self._list_resource(
                "Item Price",
                filters=[
                    ["item_code", "in", codes],
                    ["price_list", "=", price_list],
                ],
                fields=_ITEM_PRICE_FIELDS,
                limit=max(len(codes), 1),
            )
            for row in rows:
                code = str(row.get("item_code", "")).strip()
                rate = _positive_rate(row.get("price_list_rate"))
                if not code or rate is None:
                    continue
                prices[code] = {
                    "current_rate": rate,
                    "currency": row.get("currency"),
                    "uom": row.get("uom"),
                    "source": "price_list",
                    "price_list": price_list,
                }
        missing = [code for code in codes if code not in prices]
        if missing:
            rows = self._list_resource(
                "Item",
                filters=[["item_code", "in", missing]],
                fields=_ITEM_FIELDS,
                limit=max(len(missing), 1),
            )
            for row in rows:
                code = str(row.get("item_code", "")).strip()
                rate = _positive_rate(row.get("standard_rate"))
                if not code or code in prices or rate is None:
                    continue
                prices[code] = {
                    "current_rate": rate,
                    "uom": row.get("stock_uom"),
                    "source": "standard_rate",
                }
        return prices

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

    def resolve_item(self, query: str) -> dict[str, Any] | None:
        """Match by item_code first, then item_name search."""
        token = query.strip()
        if not token:
            return None
        row = self.get_item_by_code(token)
        if row:
            return row
        rows = self.search_items(token, limit=10)
        if not rows:
            return None
        lowered = token.lower()
        for row in rows:
            if str(row.get("item_name", "")).strip().lower() == lowered:
                return row
            if str(row.get("item_code", "")).strip().lower() == lowered:
                return row
        if len(rows) == 1:
            return rows[0]
        return None

    def _link_contact_to_customer(self, contact_name: str, customer_id: str) -> tuple[bool, str | None]:
        doc = self._get_resource("Contact", contact_name)
        if not str(doc.get("name", "")).strip():
            return False, "Contact not found"
        links = [link for link in (doc.get("links") or []) if isinstance(link, dict)]
        if any(
            link.get("link_doctype") == "Customer" and link.get("link_name") == customer_id
            for link in links
        ):
            return True, None
        links.append({"link_doctype": "Customer", "link_name": customer_id})
        encoded = quote(contact_name, safe="")
        result = self._put(
            f"/api/resource/Contact/{encoded}",
            json_body={"data": {"links": links}},
        )
        if result.get("_erpnext_error"):
            return False, str(result["_erpnext_error"])
        if isinstance(result.get("data"), dict):
            return True, None
        return False, "Failed to link contact to customer"

    @staticmethod
    def _split_contact_name(full_name: str) -> tuple[str, str | None]:
        parts = [part for part in full_name.split() if part]
        if not parts:
            return full_name, None
        if len(parts) == 1:
            return parts[0], None
        return parts[0], " ".join(parts[1:])

    def create_customer(
        self,
        customer_name: str,
        *,
        email: str | None = None,
        phone: str | None = None,
        company_name: str | None = None,
        customer_type: str | None = None,
        customer_group: str | None = None,
    ) -> dict[str, Any]:
        person = customer_name.strip()
        company = (company_name or "").strip()
        if not person and not company:
            return {"error": "missing_customer_name"}
        if not email and not phone:
            return {"error": "missing_identity"}

        if company:
            erp_customer_name = company
            erp_customer_type = customer_type or "Company"
            contact_first = person or company
        else:
            erp_customer_name = person
            erp_customer_type = customer_type or "Individual"
            contact_first = person

        body: dict[str, Any] = {
            "customer_name": erp_customer_name,
            "customer_type": erp_customer_type,
            "customer_group": (customer_group or "").strip() or "Individual",
        }
        customer_result = self._post_write("/api/resource/Customer", json_body={"data": body})
        if customer_result.get("_erpnext_error"):
            return {
                "error": "customer_create_failed",
                "detail": customer_result["_erpnext_error"],
            }
        customer_data = customer_result.get("data")
        if not isinstance(customer_data, dict):
            return {"error": "customer_create_failed"}
        customer_id = str(customer_data.get("name", "")).strip()
        if not customer_id:
            return {"error": "customer_create_no_name"}

        verified = self._get_resource("Customer", customer_id)
        if not str(verified.get("name", "")).strip():
            return {"error": "customer_not_verified", "customer_name": customer_id}

        first_name, last_name = self._split_contact_name(contact_first)
        existing_contact = self._find_contact(email=email, phone=phone)
        if existing_contact and existing_contact.get("name"):
            contact_name = str(existing_contact["name"]).strip()
            linked, link_error = self._link_contact_to_customer(contact_name, customer_id)
            if not linked:
                return {
                    "error": "contact_link_failed",
                    "detail": link_error,
                    "customer_name": customer_id,
                }
        else:
            contact_body: dict[str, Any] = {
                "first_name": first_name,
                "links": [{"link_doctype": "Customer", "link_name": customer_id}],
            }
            if last_name:
                contact_body["last_name"] = last_name
            if company:
                contact_body["company_name"] = company
            if email:
                contact_body[self._email_field] = email
                contact_body["email_ids"] = [{"email_id": email, "is_primary": 1}]
            if phone:
                contact_body[self._phone_field] = phone
                contact_body["phone_nos"] = [{"phone": phone, "is_primary_mobile_no": 1}]
            contact_result = self._post_write("/api/resource/Contact", json_body={"data": contact_body})
            if contact_result.get("_erpnext_error"):
                return {
                    "error": "contact_create_failed",
                    "detail": contact_result["_erpnext_error"],
                    "customer_name": customer_id,
                }
            contact_data = contact_result.get("data")
            contact_name = ""
            if isinstance(contact_data, dict):
                contact_name = str(contact_data.get("name") or "").strip()
            if not contact_name:
                return {
                    "error": "contact_create_failed",
                    "customer_name": customer_id,
                }

        encoded_customer = quote(customer_id, safe="")
        self._put(
            f"/api/resource/Customer/{encoded_customer}",
            json_body={"data": {"customer_primary_contact": contact_name}},
        )
        return {"customer_name": customer_id, "contact_name": contact_name, "company_name": company or None}

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

    def get_quotation(self, name: str) -> dict[str, Any]:
        safe = name.strip()
        if not safe:
            return {}
        encoded = quote(safe, safe="")
        result = self._get(f"/api/resource/Quotation/{encoded}")
        data = result.get("data")
        return data if isinstance(data, dict) else {}

    def submit_quotation(self, name: str) -> str | None:
        """Submit a draft quotation. Returns an error message on failure."""
        safe = name.strip()
        if not safe:
            return "Missing quotation name"

        last_error: str | None = None
        for attempt in range(2):
            doc = self.get_quotation(safe)
            if not doc:
                return f"Quotation {safe} not found"
            docstatus = doc.get("docstatus")
            if docstatus == 1:
                return None
            if docstatus == 2:
                return f"Quotation {safe} is cancelled"
            if docstatus not in (0, None):
                return f"Quotation {safe} cannot be submitted (docstatus={docstatus})"

            result = self._post_write(
                "/api/method/frappe.client.submit",
                json_body={"doc": doc},
            )
            if not result.get("_erpnext_error"):
                break
            last_error = str(result["_erpnext_error"])
            if attempt == 0 and "modified after" in last_error.lower():
                continue
            return last_error
        else:
            return last_error

        doc = self.get_quotation(safe)
        if doc.get("docstatus") != 1:
            status = doc.get("docstatus")
            return f"Quotation {safe} was not submitted (docstatus={status})"
        return None

    def _discover_quotation_print_formats(self) -> list[str]:
        rows = self._list_resource(
            "Print Format",
            filters=[["doc_type", "=", "Quotation"]],
            fields=["name"],
            limit=20,
        )
        return [str(row.get("name", "")).strip() for row in rows if str(row.get("name", "")).strip()]


def _positive_rate(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    return rate


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
            "default_price_list": doc.get("default_price_list"),
        }
    )


def _normalize_contact(doc: dict[str, Any]) -> dict[str, Any]:
    if not doc:
        return {}
    first = str(doc.get("first_name") or "").strip()
    last = str(doc.get("last_name") or "").strip()
    full_name = _contact_full_name(first, last)
    email = _contact_primary_email(doc)
    return _compact_profile(
        {
            "full_name": full_name or None,
            "first_name": first or None,
            "last_name": last if last else None,
            "email": email,
            "mobile": doc.get("mobile_no"),
            "phone": doc.get("phone"),
            "designation": doc.get("designation"),
            "department": doc.get("department"),
            "company_name": doc.get("company_name"),
        }
    )


def _contact_full_name(first: str, last: str) -> str:
    if first and last:
        return f"{first} {last}"
    return first or last


def _contact_primary_email(doc: dict[str, Any]) -> str | None:
    primary = str(doc.get("email_id") or "").strip()
    if primary:
        return primary
    child_rows = doc.get("email_ids")
    if isinstance(child_rows, list):
        for row in child_rows:
            if not isinstance(row, dict):
                continue
            email = str(row.get("email_id") or "").strip()
            if email:
                return email
    return None


def _contact_names_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = str(row.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pick_best_contact(
    docs: list[dict[str, Any]], customer: str | None
) -> dict[str, Any] | None:
    if not docs:
        return None
    if len(docs) == 1:
        return docs[0]

    def score(doc: dict[str, Any]) -> tuple[int, int, int]:
        linked = 1 if customer and _contact_links_customer(doc, customer) else 0
        person = 1 if _contact_is_person_like(doc, customer) else 0
        active = 1 if str(doc.get("status") or "").strip().lower() != "passive" else 0
        return (linked, person, active)

    return max(docs, key=score)


def _contact_links_customer(doc: dict[str, Any], customer: str) -> bool:
    links = doc.get("links")
    if not isinstance(links, list):
        return False
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("link_doctype") == "Customer" and link.get("link_name") == customer:
            return True
    return False


def _contact_is_person_like(doc: dict[str, Any], customer: str | None) -> bool:
    first = str(doc.get("first_name") or "").strip()
    last = str(doc.get("last_name") or "").strip()
    if not first:
        return False
    if customer and first == customer:
        return False
    for linked_customer in _linked_customer_names(doc):
        if first == linked_customer:
            return False
    if last:
        return True
    return first != str(doc.get("name") or "").strip()


def _linked_customer_names(doc: dict[str, Any]) -> list[str]:
    links = doc.get("links")
    if not isinstance(links, list):
        return []
    names: list[str] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("link_doctype") == "Customer" and link.get("link_name"):
            names.append(str(link["link_name"]))
    return names


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
