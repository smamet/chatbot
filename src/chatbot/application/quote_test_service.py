from __future__ import annotations

from typing import Any

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.application.customer_access_gate import can_create_quotation, resolve_manual_identity
from chatbot.application.customer_provisioning_service import (
    CustomerProvisioningError,
    ensure_erpnext_customer,
)
from chatbot.application.progress_log import ProgressLog
from chatbot.application.quote_pdf_storage import (
    TEST_QUOTE_PDF_TTL_SECONDS,
    quote_pdf_dashboard_url,
    safe_quote_filename,
    store_quote_pdf,
)
from chatbot.config.settings import Settings


def create_erpnext_quotation_for_test(
    client: ErpNextClient,
    config: dict[str, Any],
    *,
    settings: Settings,
    tenant_slug: str,
    test_email: str | None,
    test_phone: str | None,
    item_code: str,
    qty: int,
    notes: str | None = None,
    company_name: str | None = None,
    on_log: ProgressLog | None = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if on_log is not None:
            on_log.step(message)

    email, phone = resolve_manual_identity(test_email=test_email or "", test_phone=test_phone or "")
    if not email and not phone:
        return {
            "ok": False,
            "message": "Provide a test email and/or phone number.",
            "error": "missing_test_identity",
            "customer": None,
            "quote_name": None,
            "pdf_url": None,
        }
    if not can_create_quotation(config):
        return {
            "ok": False,
            "message": "Quotation creation is disabled for this connector.",
            "error": "creation_disabled",
            "customer": None,
            "quote_name": None,
            "pdf_url": None,
        }
    code = item_code.strip()
    if not code:
        return {
            "ok": False,
            "message": "Item code is required.",
            "error": "missing_item_code",
            "customer": None,
            "quote_name": None,
            "pdf_url": None,
        }
    if qty < 1:
        return {
            "ok": False,
            "message": "Quantity must be at least 1.",
            "error": "invalid_qty",
            "customer": None,
            "quote_name": None,
            "pdf_url": None,
        }

    log("Looking up customer (email / phone)…")
    customer = client.find_customer(email=email, phone=phone)
    if customer:
        log(f"Customer found: {customer}")
    else:
        log("Creating ERPNext customer…")
        try:
            customer = ensure_erpnext_customer(
                client,
                config,
                email=email,
                phone=phone,
                company_name=company_name,
            )
        except CustomerProvisioningError as exc:
            log(f"Customer creation failed: {exc}")
            return {
                "ok": False,
                "message": str(exc),
                "error": "customer_not_found",
                "customer": None,
                "quote_name": None,
                "pdf_url": None,
            }
        log(f"Customer created: {customer}")

    log(f"Resolving item “{code}”…")
    item = client.resolve_item(code)
    if not item:
        log(f"No ERPNext item matched “{code}”")
        return {
            "ok": False,
            "message": f"No ERPNext item matched “{code}”. Use item code (e.g. OKI Consumables) or exact item name.",
            "error": "item_not_found",
            "customer": customer,
            "quote_name": None,
            "pdf_url": None,
        }
    resolved_code = str(item.get("item_code", "")).strip()
    if not resolved_code:
        log("Matched item has no item_code")
        return {
            "ok": False,
            "message": "Matched item has no item_code.",
            "error": "item_not_found",
            "customer": customer,
            "quote_name": None,
            "pdf_url": None,
        }
    log(f"Item: {resolved_code}")
    rate = item.get("standard_rate")
    line: dict[str, Any] = {"item_code": resolved_code, "qty": qty}
    if rate is not None:
        line["rate"] = rate

    log("Creating ERPNext quotation…")
    created = client.create_quotation(customer, [line], notes=notes)
    quote_name = str(created.get("name", "")).strip()
    if not quote_name:
        log("ERPNext did not return a quotation name")
        return {
            "ok": False,
            "message": "ERPNext did not return a quotation name.",
            "error": "erpnext_error",
            "customer": customer,
            "quote_name": None,
            "pdf_url": None,
        }
    log(f"Quotation created: {quote_name}")

    pdf_url: str | None = None
    pdf_filename: str | None = None
    pdf_warning: str | None = None
    log("Downloading quotation PDF…")
    pdf_bytes = client.download_quotation_pdf(quote_name, on_log=on_log)
    if pdf_bytes:
        store_quote_pdf(
            settings,
            tenant_slug,
            quote_name,
            pdf_bytes,
            ttl_seconds=TEST_QUOTE_PDF_TTL_SECONDS,
        )
        pdf_filename = f"{safe_quote_filename(quote_name)}.pdf"
        pdf_url = quote_pdf_dashboard_url(tenant_slug, quote_name)
        log(f"PDF saved ({TEST_QUOTE_PDF_TTL_SECONDS}s TTL)")
    else:
        detail = client.last_pdf_error or "PDF could not be downloaded from ERPNext."
        pdf_warning = f"PDF could not be downloaded from ERPNext ({detail})."
        log(f"PDF failed: {detail}")

    return {
        "ok": True,
        "message": f"Quotation created: {quote_name}",
        "customer": customer,
        "quote_name": quote_name,
        "pdf_url": pdf_url,
        "pdf_filename": pdf_filename,
        "pdf_warning": pdf_warning,
    }
