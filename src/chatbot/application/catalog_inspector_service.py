from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from chatbot.adapters.erpnext.client import ErpNextClient, _positive_rate
from chatbot.application.erpnext_catalog_sync_service import (
    catalog_invoice_price_fallback,
    catalog_price_list,
    resolve_catalog_price,
)

_PRICE_LINE = re.compile(r"^- Price:\s*(.+)$", re.MULTILINE)
_ITEM_CODE_LINE = re.compile(r"^- Item code:\s*(.+)$", re.MULTILINE)
_ITEM_NAME_LINE = re.compile(r"^- Item name:\s*(.+)$", re.MULTILINE)
_TITLE_LINE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_DESCRIPTION_SECTION = re.compile(r"^## Description\s*\n+(.*)$", re.MULTILINE | re.DOTALL)

INVOICE_CACHE_FILENAME = ".invoice-price-cache.json"
DESCRIPTION_TRUNCATE = 120
DEFAULT_PAGE_SIZE = 50
PriceFilter = Literal["all", "with", "without"]
MismatchFilter = Literal["all", "mismatch", "aligned"]


@dataclass(frozen=True)
class RagCatalogRow:
    item_code: str
    name: str
    description: str
    price_display: str
    price_source: str | None
    price_rate: float | None
    price_currency: str | None


@dataclass(frozen=True)
class InspectorRow:
    item_code: str
    name: str
    description: str
    description_truncated: str
    rag_price_display: str
    rag_source: str | None
    item_price_display: str
    standard_rate_display: str
    invoice_price_display: str
    mismatch: bool
    expected_source: str | None


@dataclass(frozen=True)
class InvoicePriceCache:
    cached_at: str
    rates: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class InspectorPage:
    rows: list[InspectorRow]
    total: int
    page: int
    page_size: int
    total_pages: int
    stats: dict[str, int]
    invoice_cache_at: str | None
    invoice_cache_count: int
    price_list_name: str
    invoice_fallback_enabled: bool


def _format_amount(rate: float | None, currency: str | None = None) -> str:
    if rate is None:
        return "—"
    amount = int(rate) if rate == int(rate) else rate
    currency_text = str(currency or "").strip()
    return f"{amount} {currency_text}".strip() if currency_text else str(amount)


def _parse_price_line(raw: str) -> tuple[str, str | None, float | None, str | None]:
    text = raw.strip()
    if not text or text.lower() == "not available":
        return "not available", None, None, None
    match = re.match(r"^(.+?)\s+\(([^)]+)\)\s*$", text)
    if match:
        amount_part, source = match.group(1).strip(), match.group(2).strip()
    else:
        amount_part, source = text, "standard_rate"
    parts = amount_part.split(None, 1)
    if not parts:
        return text, source, None, None
    try:
        rate = float(parts[0].replace(",", ""))
    except ValueError:
        return text, source, None, None
    if rate <= 0:
        return "not available", None, None, None
    currency = parts[1].strip() if len(parts) > 1 else None
    display = _format_amount(rate, currency)
    if source and source != "standard_rate":
        display = f"{display} ({source})"
    return display, source or None, rate, currency


def parse_catalog_markdown(content: str) -> RagCatalogRow | None:
    code_match = _ITEM_CODE_LINE.search(content)
    if not code_match:
        return None
    item_code = code_match.group(1).strip()
    if not item_code:
        return None
    title_match = _TITLE_LINE.search(content)
    name_match = _ITEM_NAME_LINE.search(content)
    name = (name_match.group(1).strip() if name_match else "") or (
        title_match.group(1).strip() if title_match else item_code
    )
    desc_match = _DESCRIPTION_SECTION.search(content)
    description = desc_match.group(1).strip() if desc_match else ""
    price_match = _PRICE_LINE.search(content)
    price_raw = price_match.group(1).strip() if price_match else "not available"
    price_display, price_source, price_rate, price_currency = _parse_price_line(price_raw)
    return RagCatalogRow(
        item_code=item_code,
        name=name,
        description=description,
        price_display=price_display,
        price_source=price_source,
        price_rate=price_rate,
        price_currency=price_currency,
    )


def load_rag_rows(catalog_dir: Path) -> dict[str, RagCatalogRow]:
    if not catalog_dir.is_dir():
        return {}
    rows: dict[str, RagCatalogRow] = {}
    for path in sorted(catalog_dir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        row = parse_catalog_markdown(content)
        if row is not None:
            rows[row.item_code] = row
    return rows


def invoice_cache_path(catalog_dir: Path) -> Path:
    return catalog_dir / INVOICE_CACHE_FILENAME


def read_invoice_cache(catalog_dir: Path) -> InvoicePriceCache | None:
    path = invoice_cache_path(catalog_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cached_at = str(payload.get("cached_at") or "").strip()
    rates = payload.get("rates")
    if not cached_at or not isinstance(rates, dict):
        return None
    clean_rates: dict[str, dict[str, Any]] = {}
    for code, entry in rates.items():
        if isinstance(entry, dict) and str(code).strip():
            clean_rates[str(code).strip()] = entry
    return InvoicePriceCache(cached_at=cached_at, rates=clean_rates)


def write_invoice_cache(catalog_dir: Path, rates: dict[str, dict[str, Any]]) -> InvoicePriceCache:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    cached_at = datetime.now(UTC).isoformat()
    payload = {"cached_at": cached_at, "rates": rates}
    invoice_cache_path(catalog_dir).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return InvoicePriceCache(cached_at=cached_at, rates=rates)


def fetch_and_cache_invoice_rates(
    catalog_dir: Path,
    client: ErpNextClient,
) -> InvoicePriceCache:
    rates = client.fetch_latest_invoice_rates()
    return write_invoice_cache(catalog_dir, rates)


def fetch_live_erp_columns(
    client: ErpNextClient,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    price_list = catalog_price_list(config)
    item_prices = client.fetch_price_list_rates(price_list)
    standard_rates: dict[str, float] = {}
    for item in client.list_catalog_items():
        code = str(item.get("item_code", "")).strip()
        rate = _positive_rate(item.get("standard_rate"))
        if code and rate is not None:
            standard_rates[code] = rate
    return item_prices, standard_rates


def _truncate(text: str, limit: int = DESCRIPTION_TRUNCATE) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _expected_price_entry(
    item_code: str,
    *,
    item_prices: dict[str, dict[str, Any]],
    standard_rates: dict[str, float],
    invoice_cache: InvoicePriceCache | None,
    config: dict[str, Any],
) -> tuple[str, str | None, float | None]:
    item_stub = {"item_code": item_code, "standard_rate": standard_rates.get(item_code)}
    price_entry = item_prices.get(item_code)
    if price_entry is None and catalog_invoice_price_fallback(config):
        cached = (invoice_cache.rates.get(item_code) if invoice_cache else None) or None
        if cached:
            rate = _positive_rate(cached.get("rate"))
            if rate is not None:
                price_entry = {
                    "rate": rate,
                    "currency": cached.get("currency"),
                    "price_list": "last invoice",
                }
    amount_str, source = resolve_catalog_price(item_stub, price_entry)
    if amount_str is None:
        return "not available", None, None
    display = f"{amount_str} ({source})" if source else amount_str
    rate: float | None = None
    if price_entry and price_entry.get("rate") is not None:
        rate = _positive_rate(price_entry.get("rate"))
    elif source is None:
        rate = _positive_rate(item_stub.get("standard_rate"))
    return display, source, rate


def _rates_match(a: float | None, b: float | None, *, tolerance: float = 0.01) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < tolerance


def merge_inspector_rows(
    rag_rows: dict[str, RagCatalogRow],
    *,
    item_prices: dict[str, dict[str, Any]],
    standard_rates: dict[str, float],
    invoice_cache: InvoicePriceCache | None,
    config: dict[str, Any],
) -> list[InspectorRow]:
    merged: list[InspectorRow] = []
    for code in sorted(rag_rows):
        rag = rag_rows[code]
        price_entry = item_prices.get(code)
        item_price_display = "—"
        if price_entry:
            amount_str, source = resolve_catalog_price(
                {"item_code": code, "standard_rate": standard_rates.get(code)},
                price_entry,
            )
            if amount_str:
                item_price_display = f"{amount_str} ({source})" if source else amount_str
        std_rate = standard_rates.get(code)
        standard_rate_display = _format_amount(std_rate)
        invoice_price_display = "—"
        if invoice_cache and code in invoice_cache.rates:
            cached = invoice_cache.rates[code]
            rate = _positive_rate(cached.get("rate"))
            if rate is not None:
                invoice_price_display = _format_amount(rate, str(cached.get("currency") or ""))
        expected_display, expected_source, expected_rate = _expected_price_entry(
            code,
            item_prices=item_prices,
            standard_rates=standard_rates,
            invoice_cache=invoice_cache,
            config=config,
        )
        mismatch = not _rates_match(rag.price_rate, expected_rate)
        if rag.price_display == "not available" and expected_display == "not available":
            mismatch = False
        merged.append(
            InspectorRow(
                item_code=code,
                name=rag.name,
                description=rag.description,
                description_truncated=_truncate(rag.description),
                rag_price_display=rag.price_display,
                rag_source=rag.price_source,
                item_price_display=item_price_display,
                standard_rate_display=standard_rate_display,
                invoice_price_display=invoice_price_display,
                mismatch=mismatch,
                expected_source=expected_source,
            )
        )
    return merged


def filter_rows(rows: list[InspectorRow], query: str) -> list[InspectorRow]:
    q = query.strip().lower()
    if not q:
        return rows
    filtered: list[InspectorRow] = []
    for row in rows:
        haystack = " ".join(
            [
                row.item_code,
                row.name,
                row.description,
                row.rag_price_display,
            ]
        ).lower()
        if q in haystack:
            filtered.append(row)
    return filtered


def normalize_price_filter(value: str) -> PriceFilter:
    normalized = (value or "all").strip().lower()
    if normalized in ("with", "without"):
        return normalized
    return "all"


def filter_by_rag_price(
    rows: list[InspectorRow],
    price_filter: str,
) -> list[InspectorRow]:
    normalized = normalize_price_filter(price_filter)
    if normalized == "with":
        return [row for row in rows if row.rag_price_display != "not available"]
    if normalized == "without":
        return [row for row in rows if row.rag_price_display == "not available"]
    return rows


def normalize_mismatch_filter(value: str) -> MismatchFilter:
    normalized = (value or "all").strip().lower()
    if normalized in ("mismatch", "aligned"):
        return normalized
    return "all"


def filter_by_mismatch(
    rows: list[InspectorRow],
    mismatch_filter: str,
) -> list[InspectorRow]:
    normalized = normalize_mismatch_filter(mismatch_filter)
    if normalized == "mismatch":
        return [row for row in rows if row.mismatch]
    if normalized == "aligned":
        return [row for row in rows if not row.mismatch]
    return rows


def paginate_rows(
    rows: list[InspectorRow],
    *,
    page: int,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> InspectorPage:
    size = max(1, page_size)
    current = max(1, page)
    total = len(rows)
    total_pages = max(1, (total + size - 1) // size)
    if current > total_pages:
        current = total_pages
    start = (current - 1) * size
    page_rows = rows[start : start + size]
    mismatches = sum(1 for row in rows if row.mismatch)
    with_rag_price = sum(1 for row in rows if row.rag_price_display != "not available")
    return InspectorPage(
        rows=page_rows,
        total=total,
        page=current,
        page_size=size,
        total_pages=total_pages,
        stats={
            "total": total,
            "with_rag_price": with_rag_price,
            "without_rag_price": total - with_rag_price,
            "mismatches": mismatches,
        },
        invoice_cache_at=None,
        invoice_cache_count=0,
        price_list_name="",
        invoice_fallback_enabled=False,
    )


def build_inspector_page(
    catalog_dir: Path,
    *,
    client: ErpNextClient,
    config: dict[str, Any],
    query: str = "",
    price_filter: str = "all",
    mismatch_filter: str = "all",
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> InspectorPage:
    rag_rows = load_rag_rows(catalog_dir)
    item_prices, standard_rates = fetch_live_erp_columns(client, config)
    invoice_cache = read_invoice_cache(catalog_dir)
    rows = merge_inspector_rows(
        rag_rows,
        item_prices=item_prices,
        standard_rates=standard_rates,
        invoice_cache=invoice_cache,
        config=config,
    )
    rows = filter_rows(rows, query)
    rows = filter_by_rag_price(rows, price_filter)
    rows = filter_by_mismatch(rows, mismatch_filter)
    result = paginate_rows(rows, page=page, page_size=page_size)
    return InspectorPage(
        rows=result.rows,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
        stats=result.stats,
        invoice_cache_at=invoice_cache.cached_at if invoice_cache else None,
        invoice_cache_count=len(invoice_cache.rates) if invoice_cache else 0,
        price_list_name=catalog_price_list(config),
        invoice_fallback_enabled=catalog_invoice_price_fallback(config),
    )


def inspector_row_to_dict(row: InspectorRow) -> dict[str, Any]:
    return {
        "item_code": row.item_code,
        "name": row.name,
        "description": row.description,
        "description_truncated": row.description_truncated,
        "rag_price_display": row.rag_price_display,
        "rag_source": row.rag_source,
        "item_price_display": row.item_price_display,
        "standard_rate_display": row.standard_rate_display,
        "invoice_price_display": row.invoice_price_display,
        "mismatch": row.mismatch,
        "expected_source": row.expected_source,
    }


def inspector_page_to_dict(page: InspectorPage) -> dict[str, Any]:
    return {
        "rows": [inspector_row_to_dict(row) for row in page.rows],
        "total": page.total,
        "page": page.page,
        "page_size": page.page_size,
        "total_pages": page.total_pages,
        "stats": page.stats,
        "invoice_cache_at": page.invoice_cache_at,
        "invoice_cache_count": page.invoice_cache_count,
        "price_list_name": page.price_list_name,
        "invoice_fallback_enabled": page.invoice_fallback_enabled,
    }
