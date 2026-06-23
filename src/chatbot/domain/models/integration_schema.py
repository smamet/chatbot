from __future__ import annotations

from dataclasses import dataclass

from chatbot.domain.models.integration import IntegrationType


@dataclass(frozen=True)
class IntegrationField:
    key: str
    label: str
    help: str
    input_type: str = "text"
    required: bool = False
    placeholder: str = ""
    secret: bool = False
    default: str = ""
    options: tuple[tuple[str, str], ...] | None = None
    oauth_managed: bool = False


@dataclass(frozen=True)
class IntegrationMeta:
    label: str
    description: str


INTEGRATION_META: dict[str, IntegrationMeta] = {
    IntegrationType.ERPNEXT.value: IntegrationMeta(
        label="ERPNext",
        description="Lookup customers by channel identity and inject recent orders and quotations.",
    ),
    IntegrationType.QUICKBOOKS.value: IntegrationMeta(
        label="QuickBooks Online",
        description="Connect via Intuit OAuth and enrich replies with invoices and estimates.",
    ),
}

INTEGRATION_SCHEMAS: dict[str, list[IntegrationField]] = {
    IntegrationType.ERPNEXT.value: [
        IntegrationField(
            key="url",
            label="ERPNext URL",
            help="Base URL of your ERPNext site (e.g. https://erp.example.com).",
            placeholder="https://erp.example.com",
            required=True,
        ),
        IntegrationField(
            key="api_key",
            label="API key",
            help="ERPNext API key (User > API Access). Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            required=True,
        ),
        IntegrationField(
            key="api_secret",
            label="API secret",
            help="ERPNext API secret paired with the key. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            required=True,
        ),
        IntegrationField(
            key="identity_email_field",
            label="Contact email field",
            help="ERPNext Contact field matched against inbound email addresses.",
            default="email_id",
        ),
        IntegrationField(
            key="identity_phone_field",
            label="Contact phone field",
            help="ERPNext Contact field matched against WhatsApp phone numbers.",
            default="mobile_no",
        ),
        IntegrationField(
            key="fetch_orders",
            label="Fetch sales invoices",
            help="Include recent Sales Invoices (with line items) in the customer context injected into the bot prompt.",
            input_type="checkbox",
            default="true",
        ),
        IntegrationField(
            key="fetch_quotations",
            label="Fetch quotations",
            help="Include recent Quotations (with line items) in the customer context injected into the bot prompt.",
            input_type="checkbox",
            default="true",
        ),
        IntegrationField(
            key="max_items",
            label="Max items per list",
            help="Maximum number of orders and quotations returned per customer.",
            input_type="number",
            default="5",
        ),
        IntegrationField(
            key="fetch_current_prices",
            label="Fetch current prices",
            help="Bulk lookup of Item Price for products on recent invoices/quotes. Uses the customer's default price list; no discount rules.",
            input_type="checkbox",
            default="true",
        ),
        IntegrationField(
            key="default_customer_group",
            label="Default customer group",
            help="Leaf Customer Group in ERPNext (not a parent group like “All Customer Groups”). Common values: Individual, Commercial.",
            placeholder="Individual",
            default="Individual",
        ),
        IntegrationField(
            key="allow_create_customer",
            label="Allow customer creation",
            help="Create a new ERPNext Customer (+ Contact) when none matches the channel identity.",
            input_type="checkbox",
            default="false",
        ),
        IntegrationField(
            key="allow_create_quotation",
            label="Allow quotation creation",
            help="Create ERPNext Quotations when a quote hook is approved. Disable for read-only connectors.",
            input_type="checkbox",
            default="false",
        ),
        IntegrationField(
            key="quotation_print_format",
            label="Quotation print format",
            help="Optional ERPNext Print Format name for quotation PDFs (e.g. Vdtec Quotation). Leave blank to use the ERPNext default.",
            placeholder="Vdtec Quotation",
            default="",
        ),
        IntegrationField(
            key="sync_catalog_to_rag",
            label="Sync catalog to RAG",
            help="Export ERPNext items (price, stock) into the bot knowledge base on a schedule.",
            input_type="checkbox",
            default="false",
        ),
        IntegrationField(
            key="catalog_sync_interval_minutes",
            label="Catalog sync interval (minutes)",
            help="How often to refresh the ERPNext catalog snapshot in RAG.",
            input_type="number",
            default="360",
        ),
        IntegrationField(
            key="catalog_include_stock",
            label="Include stock totals",
            help="Aggregate Bin quantities across warehouses into each catalog item file.",
            input_type="checkbox",
            default="true",
        ),
        IntegrationField(
            key="catalog_price_list",
            label="Catalog price list",
            help="ERPNext selling price list for catalog RAG (e.g. Standard Selling). Defaults to Standard Selling when blank.",
            placeholder="Standard Selling",
            default="Standard Selling",
        ),
        IntegrationField(
            key="catalog_invoice_price_fallback",
            label="Invoice price fallback (catalog sync)",
            help="When syncing the catalog to RAG: if Item Price and standard rate are missing, use the latest submitted Sales Invoice line rate. Off by default.",
            input_type="checkbox",
            default="false",
        ),
    ],
    IntegrationType.QUICKBOOKS.value: [
        IntegrationField(
            key="environment",
            label="Environment",
            help="Sandbox for development, Production for live QuickBooks companies.",
            input_type="select",
            default="sandbox",
            options=(("sandbox", "Sandbox"), ("production", "Production")),
        ),
        IntegrationField(
            key="client_id",
            label="Client ID",
            help="Intuit app Client ID. Leave blank on update to keep current value.",
            secret=True,
            required=True,
        ),
        IntegrationField(
            key="client_secret",
            label="Client Secret",
            help="Intuit app Client Secret. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            required=True,
        ),
        IntegrationField(
            key="fetch_invoices",
            label="Fetch invoices",
            help="Include recent Invoices in the customer context injected into the bot prompt.",
            input_type="checkbox",
            default="true",
        ),
        IntegrationField(
            key="fetch_estimates",
            label="Fetch estimates",
            help="Include recent Estimates in the customer context injected into the bot prompt.",
            input_type="checkbox",
            default="true",
        ),
        IntegrationField(
            key="max_items",
            label="Max items per list",
            help="Maximum number of invoices and estimates returned per customer.",
            input_type="number",
            default="5",
        ),
        IntegrationField(
            key="realm_id",
            label="Realm ID",
            help="Set automatically after OAuth connect.",
            oauth_managed=True,
        ),
        IntegrationField(
            key="access_token",
            label="Access token",
            help="Set automatically after OAuth connect.",
            secret=True,
            oauth_managed=True,
        ),
        IntegrationField(
            key="refresh_token",
            label="Refresh token",
            help="Set automatically after OAuth connect.",
            secret=True,
            oauth_managed=True,
        ),
        IntegrationField(
            key="token_expires_at",
            label="Token expiry",
            help="Set automatically after OAuth connect.",
            oauth_managed=True,
        ),
    ],
}


def fields_for(integration_type: str, *, include_oauth_managed: bool = False) -> list[IntegrationField]:
    result: list[IntegrationField] = []
    for field in INTEGRATION_SCHEMAS.get(integration_type, []):
        if field.oauth_managed and not include_oauth_managed:
            continue
        result.append(field)
    return result


def secret_integration_keys() -> frozenset[str]:
    return frozenset(
        field.key
        for fields in INTEGRATION_SCHEMAS.values()
        for field in fields
        if field.secret
    )


def integration_meta_for_template() -> dict[str, dict[str, str]]:
    return {
        integration_type: {"label": meta.label, "description": meta.description}
        for integration_type, meta in INTEGRATION_META.items()
    }


def integration_schemas_for_template() -> dict[str, list[dict]]:
    return {
        integration_type: [
            {
                "key": field.key,
                "label": field.label,
                "help": field.help,
                "input_type": field.input_type,
                "required": field.required,
                "placeholder": field.placeholder or field.key,
                "default": field.default,
                "options": [{"value": v, "label": lbl} for v, lbl in field.options]
                if field.options
                else None,
                "oauth_managed": field.oauth_managed,
            }
            for field in fields_for(integration_type)
        ]
        for integration_type in INTEGRATION_SCHEMAS
    }


def is_quickbooks_connected(config: dict) -> bool:
    return bool(str(config.get("realm_id", "")).strip() and str(config.get("refresh_token", "")).strip())
