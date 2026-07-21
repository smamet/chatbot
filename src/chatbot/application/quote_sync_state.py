from __future__ import annotations

import re
from typing import Any

from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.application.erpnext_error_display import format_erpnext_error_message
from chatbot.application.quote_pdf_storage import quote_pdf_dashboard_url

_ERPNEXT_DATETIME_MICROSECONDS = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+"
)


def normalize_erpnext_modified(value: str | None) -> str | None:
    if not value:
        return None
    return _ERPNEXT_DATETIME_MICROSECONDS.sub(r"\1", value.strip())


def quotation_erp_modified(client: ErpNextClient, quote_name: str) -> str | None:
    doc = client.get_quotation(quote_name.strip())
    modified = doc.get("modified")
    if modified is None:
        return None
    return str(modified).strip() or None


def quote_pdf_is_stale(stored_modified: str | None, current_modified: str | None) -> bool:
    if not stored_modified or not current_modified:
        return False
    return normalize_erpnext_modified(stored_modified) != normalize_erpnext_modified(
        current_modified
    )


def quote_pdf_stale_context(
    *,
    client: ErpNextClient | None,
    tenant_slug: str,
    quote_name: str | None,
    stored_modified: str | None,
    erpnext_url: str | None,
) -> dict[str, Any]:
    if not quote_name or client is None:
        return {
            "stale": False,
            "stored_modified": stored_modified,
            "current_modified": None,
            "download_url": None,
            "erpnext_url": erpnext_url,
        }
    current_modified = quotation_erp_modified(client, quote_name)
    stale = quote_pdf_is_stale(stored_modified, current_modified)
    return {
        "stale": stale,
        "stored_modified": format_erpnext_error_message(stored_modified),
        "current_modified": format_erpnext_error_message(current_modified),
        "download_url": f"{quote_pdf_dashboard_url(tenant_slug, quote_name)}?inline=1",
        "erpnext_url": erpnext_url,
    }
