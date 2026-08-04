from __future__ import annotations

from typing import Any

from evenor.adapters.erpnext.client import ErpNextClient
from evenor.application.customer_access_gate import can_create_customer


class CustomerProvisioningError(RuntimeError):
    pass


def _default_name_from_identity(email: str | None, phone: str | None) -> str:
    if email:
        local = email.split("@", 1)[0].strip()
        cleaned = local.replace(".", " ").replace("_", " ").strip()
        return cleaned.title() if cleaned else email
    if phone:
        return phone
    return "Customer"


def ensure_erpnext_customer(
    client: ErpNextClient,
    config: dict[str, Any],
    *,
    email: str | None,
    phone: str | None,
    customer_name: str | None = None,
    company_name: str | None = None,
) -> str:
    existing = client.find_customer(email=email, phone=phone)
    if existing:
        return existing
    if not can_create_customer(config):
        raise CustomerProvisioningError("Customer creation is disabled for this connector")
    if not email and not phone:
        raise CustomerProvisioningError("Email or phone is required to create a customer")
    person = (customer_name or "").strip() or _default_name_from_identity(email, phone)
    company = (company_name or "").strip() or None
    group = str(config.get("default_customer_group", "")).strip() or None
    created = client.create_customer(
        person,
        email=email,
        phone=phone,
        company_name=company,
        customer_group=group,
    )
    if created.get("error"):
        error = str(created.get("error", "unknown"))
        detail = str(created.get("detail", "")).strip()
        message = f"ERPNext customer creation failed ({error})"
        if detail:
            message = f"{message}: {detail}"
        raise CustomerProvisioningError(message)
    customer_id = str(created.get("customer_name", "")).strip()
    if not customer_id:
        raise CustomerProvisioningError("ERPNext did not return a customer name")
    return customer_id


def create_erpnext_customer_for_test(
    client: ErpNextClient,
    config: dict[str, Any],
    *,
    test_email: str | None,
    test_phone: str | None,
    customer_name: str | None = None,
    company_name: str | None = None,
) -> dict[str, Any]:
    from evenor.application.customer_access_gate import resolve_manual_identity

    email, phone = resolve_manual_identity(test_email=test_email or "", test_phone=test_phone or "")
    if not email and not phone:
        return {
            "ok": False,
            "message": "Provide a test email and/or phone number.",
            "error": "missing_test_identity",
            "customer": None,
            "created": False,
        }
    if not can_create_customer(config):
        return {
            "ok": False,
            "message": "Customer creation is disabled for this connector.",
            "error": "creation_disabled",
            "customer": None,
            "created": False,
        }
    existing = client.find_customer(email=email, phone=phone)
    if existing:
        return {
            "ok": True,
            "message": f"Customer already exists: {existing}",
            "customer": existing,
            "created": False,
        }
    try:
        person = (customer_name or "").strip() or _default_name_from_identity(email, phone)
        company = (company_name or "").strip() or None
        group = str(config.get("default_customer_group", "")).strip() or None
        created = client.create_customer(
            person,
            email=email,
            phone=phone,
            company_name=company,
            customer_group=group,
        )
        if created.get("error"):
            error = str(created.get("error", "unknown"))
            detail = str(created.get("detail", "")).strip()
            message = f"ERPNext customer creation failed ({error})"
            if detail:
                message = f"{message}: {detail}"
            raise CustomerProvisioningError(message)
        customer = str(created.get("customer_name", "")).strip()
        if not customer:
            raise CustomerProvisioningError("ERPNext did not return a customer name")
    except CustomerProvisioningError as exc:
        return {
            "ok": False,
            "message": str(exc),
            "error": "provisioning_failed",
            "customer": None,
            "created": False,
        }
    return {
        "ok": True,
        "message": (
            f"Customer created: {customer}"
            + (f" (company: {created.get('company_name')})" if created.get("company_name") else "")
            + f" (contact: {created.get('contact_name', '—')})"
        ),
        "customer": customer,
        "company": created.get("company_name"),
        "contact": created.get("contact_name"),
        "created": True,
    }
