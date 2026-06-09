from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.erpnext.client import ErpNextClient, _positive_rate
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.tenant_paths import safe_catalog_filename, tenant_catalog_dir
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.sync_service import IngestSyncService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.application.tenant_service import TenantService
from chatbot.config.settings import Settings
from chatbot.domain.models.integration import IntegrationType


@dataclass
class CatalogSyncResult:
    ok: bool
    message: str
    item_count: int = 0
    files_written: int = 0
    files_removed: int = 0
    logs: list[str] = field(default_factory=list)


def catalog_sync_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("sync_catalog_to_rag"))


def catalog_sync_interval_minutes(config: dict[str, Any]) -> int:
    try:
        value = int(config.get("catalog_sync_interval_minutes", 360))
    except (TypeError, ValueError):
        value = 360
    return max(1, value)


def catalog_include_stock(config: dict[str, Any]) -> bool:
    value = config.get("catalog_include_stock")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "on", "yes")


def catalog_price_list(config: dict[str, Any]) -> str:
    if "catalog_price_list" not in config:
        return "Standard Selling"
    return str(config.get("catalog_price_list", "")).strip()


def resolve_catalog_price(
    item: dict[str, Any],
    price_entry: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if price_entry:
        rate = price_entry.get("rate")
        if rate is not None:
            amount = int(rate) if rate == int(rate) else rate
            currency = str(price_entry.get("currency") or "").strip()
            amount_str = f"{amount} {currency}".strip() if currency else str(amount)
            return amount_str, str(price_entry.get("price_list") or "").strip() or None
    standard = _positive_rate(item.get("standard_rate"))
    if standard is not None:
        amount = int(standard) if standard == int(standard) else standard
        return str(amount), None
    return None, None


def format_price_line(item: dict[str, Any], price_entry: dict[str, Any] | None) -> str:
    amount_str, source = resolve_catalog_price(item, price_entry)
    if amount_str is None:
        return "- Price: not available"
    if source:
        return f"- Price: {amount_str} ({source})"
    return f"- Price: {amount_str}"


def render_item_markdown(
    item: dict[str, Any],
    *,
    stock_qty: float | None,
    sync_date: str,
    include_stock: bool,
    price_entry: dict[str, Any] | None = None,
) -> str:
    code = str(item.get("item_code", "")).strip()
    name = str(item.get("item_name", "")).strip()
    group = str(item.get("item_group", "")).strip()
    uom = str(item.get("stock_uom", "")).strip()
    description = str(item.get("description", "")).strip()
    lines = [
        f"# {name or code}",
        "",
        f"- Item code: {code}",
    ]
    if name and name != code:
        lines.append(f"- Item name: {name}")
    if group:
        lines.append(f"- Item group: {group}")
    lines.append(format_price_line(item, price_entry))
    if uom:
        lines.append(f"- UOM: {uom}")
    if include_stock and stock_qty is not None:
        qty_text = int(stock_qty) if stock_qty == int(stock_qty) else stock_qty
        lines.append(f"- Total stock: {qty_text} {uom}".rstrip())
    lines.append(f"- Stock/price as of: {sync_date}")
    if description:
        lines.extend(["", "## Description", "", description])
    return "\n".join(lines).strip() + "\n"


def sync_catalog_files(
    settings: Settings,
    slug: str,
    items: list[dict[str, Any]],
    stock_totals: dict[str, float],
    *,
    include_stock: bool,
    price_by_code: dict[str, dict[str, Any]] | None = None,
    sync_date: str | None = None,
) -> tuple[int, int, list[str]]:
    catalog_root = tenant_catalog_dir(settings, slug)
    sync_label = sync_date or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    logs: list[str] = []
    written = 0
    active_paths: set[Path] = set()
    prices = price_by_code or {}

    for item in items:
        code = str(item.get("item_code", "")).strip()
        if not code:
            continue
        safe = safe_catalog_filename(code)
        path = catalog_root / f"{safe}.md"
        active_paths.add(path)
        stock_qty = stock_totals.get(code) if include_stock else None
        content = render_item_markdown(
            item,
            stock_qty=stock_qty,
            sync_date=sync_label,
            include_stock=include_stock,
            price_entry=prices.get(code),
        )
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            continue
        path.write_text(content, encoding="utf-8")
        written += 1
        logs.append(f"wrote: {path.name}")

    removed = 0
    for path in catalog_root.glob("*.md"):
        if path not in active_paths:
            path.unlink(missing_ok=True)
            removed += 1
            logs.append(f"removed: {path.name}")

    return written, removed, logs


def reconcile_catalog_rag(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    slug: str,
    batch_size: int = 100,
    pause_seconds: float = 0.0,
) -> list[str]:
    merged = settings
    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    tenant = tenant_svc.get_by_slug(slug)
    if tenant is not None:
        merged = merge_tenant_settings(settings, tenant)

    catalog_root = tenant_catalog_dir(settings, slug)
    store = LanceVectorStore(settings.lancedb_root / slug)
    embedder = GeminiEmbedder()
    sync_svc = IngestSyncService(
        settings=merged,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant_id,
    )
    logs = sync_svc.prune_missing_under_root(catalog_root)
    md_files = sorted(catalog_root.glob("*.md"))
    logs.extend(
        sync_svc.ingest_paths_batched(
            md_files,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
        )
    )
    return logs


def sync_erpnext_catalog_for_tenant(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    tenant_slug: str,
    config: dict[str, Any],
    client: ErpNextClient | None = None,
    batch_size: int = 100,
    pause_seconds: float = 0.0,
) -> CatalogSyncResult:
    erp_client = client or ErpNextClient(config)
    include_stock = catalog_include_stock(config)
    sync_date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    try:
        items = erp_client.list_catalog_items()
        stock_totals = erp_client.fetch_stock_totals() if include_stock else {}
        price_list_name = catalog_price_list(config)
        price_by_code = (
            erp_client.fetch_price_list_rates(price_list_name) if price_list_name else {}
        )
    except Exception as exc:
        return CatalogSyncResult(ok=False, message=str(exc))

    written, removed, file_logs = sync_catalog_files(
        settings,
        tenant_slug,
        items,
        stock_totals,
        include_stock=include_stock,
        price_by_code=price_by_code,
        sync_date=sync_date,
    )
    try:
        rag_logs = reconcile_catalog_rag(
            session,
            settings=settings,
            tenant_id=tenant_id,
            slug=tenant_slug,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
        )
    except Exception as exc:
        return CatalogSyncResult(
            ok=False,
            message=str(exc),
            item_count=len(items),
            files_written=written,
            files_removed=removed,
            logs=file_logs,
        )

    return CatalogSyncResult(
        ok=True,
        message=f"Catalog synced: {len(items)} items",
        item_count=len(items),
        files_written=written,
        files_removed=removed,
        logs=file_logs + rag_logs,
    )


def update_catalog_sync_metadata(
    session: Session,
    integration_id: int,
    *,
    result: CatalogSyncResult,
) -> None:
    repo = SqlAlchemyIntegrationRepository(session)
    integration = repo.find_by_id(integration_id)
    if integration is None:
        return
    merged = dict(integration.config)
    merged["catalog_last_sync_at"] = datetime.now(UTC).isoformat()
    merged["catalog_last_item_count"] = result.item_count
    merged["catalog_last_error"] = None if result.ok else result.message
    repo.update(integration_id, config=merged)


def catalog_sync_due(config: dict[str, Any], *, now: datetime | None = None) -> bool:
    if not catalog_sync_enabled(config):
        return False
    last_raw = str(config.get("catalog_last_sync_at", "")).strip()
    if not last_raw:
        return True
    try:
        last = datetime.fromisoformat(last_raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    elapsed_minutes = (current - last).total_seconds() / 60.0
    return elapsed_minutes >= catalog_sync_interval_minutes(config)


def run_due_catalog_syncs(
    session: Session,
    *,
    settings: Settings,
    batch_size: int = 100,
    pause_seconds: float = 0.0,
) -> list[str]:
    repo = SqlAlchemyIntegrationRepository(session)
    tenant_repo = SqlAlchemyTenantRepository(session)
    tenant_svc = TenantService(tenant_repo)
    logs: list[str] = []

    rows = repo.list_active_by_type(IntegrationType.ERPNEXT)
    for integration in rows:
        if not catalog_sync_due(integration.config):
            continue
        tenant = tenant_svc.get_by_id(integration.tenant_id)
        if tenant is None:
            continue
        result = sync_erpnext_catalog_for_tenant(
            session,
            settings=settings,
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            config=integration.config,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
        )
        update_catalog_sync_metadata(session, integration.id, result=result)
        session.commit()
        status = "ok" if result.ok else "failed"
        logs.append(f"{tenant.slug}: {status} — {result.message}")
    return logs
