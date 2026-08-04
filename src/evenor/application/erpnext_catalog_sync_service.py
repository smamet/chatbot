from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from evenor.adapters.embeddings.gemini_embedder import GeminiEmbedder
from evenor.adapters.erpnext.client import ErpNextClient, _positive_rate
from evenor.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from evenor.adapters.persistence.orm import IngestedFileRow
from evenor.adapters.persistence.tenant_paths import safe_catalog_filename, tenant_catalog_dir
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.adapters.rag.lance_vector_store import LanceVectorStore
from evenor.application.ingest_service import file_content_hash
from evenor.application.fx_rate_service import FxRateService
from evenor.application.sync_service import IngestSyncService, _is_path_under_root
from evenor.application.tenant_settings import merge_tenant_settings
from evenor.application.usage_metering import metered_embedder
from evenor.application.tenant_service import TenantService
from evenor.config.settings import Settings
from evenor.domain.models.integration import IntegrationType


@dataclass
class CatalogSyncResult:
    ok: bool
    message: str
    item_count: int = 0
    files_written: int = 0
    files_removed: int = 0
    rag_files_indexed: int = 0
    logs: list[str] = field(default_factory=list)


@dataclass
class CatalogRagIndexPlan:
    needs_embed: list[Path]
    already_indexed: list[Path]


@dataclass
class CatalogFileSyncResult:
    written: int
    removed: int
    logs: list[str]
    changed_paths: list[Path] = field(default_factory=list)
    removed_paths: list[Path] = field(default_factory=list)


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
    raw = config.get("catalog_price_list", "Standard Selling")
    name = str(raw).strip()
    return name or "Standard Selling"


def catalog_invoice_price_fallback(config: dict[str, Any]) -> bool:
    if catalog_use_highest_price(config):
        return False
    value = config.get("catalog_invoice_price_fallback")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "on", "yes")


def catalog_use_highest_price(config: dict[str, Any]) -> bool:
    value = config.get("catalog_use_highest_price")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "on", "yes")


def catalog_price_compare_currency(config: dict[str, Any]) -> str:
    raw = config.get("catalog_price_compare_currency", "MUR")
    return str(raw).strip().upper() or "MUR"


def pick_catalog_price_entry(
    *,
    item_price_entry: dict[str, Any] | None,
    invoice_entry: dict[str, Any] | None,
    standard_rate: float | None,
    fx: FxRateService | None,
    compare_base: str,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if item_price_entry and _positive_rate(item_price_entry.get("rate")) is not None:
        candidates.append(item_price_entry)
    if invoice_entry and _positive_rate(invoice_entry.get("rate")) is not None:
        candidates.append(invoice_entry)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    compare_ccy = compare_base.strip().upper() or "MUR"
    best: dict[str, Any] | None = None
    best_converted = -1.0
    for entry in candidates:
        rate = _positive_rate(entry.get("rate"))
        if rate is None:
            continue
        currency = str(entry.get("currency") or compare_ccy).strip().upper() or compare_ccy
        converted = rate if currency == compare_ccy else None
        if converted is None and fx is not None:
            converted = fx.convert(rate, currency, compare_ccy)
        if converted is None:
            continue
        if converted > best_converted:
            best_converted = converted
            best = entry
    if best is not None:
        return best
    return item_price_entry if item_price_entry in candidates else candidates[0]


def build_catalog_price_map(
    items: list[dict[str, Any]],
    *,
    erp_client: ErpNextClient,
    config: dict[str, Any],
    fx: FxRateService | None = None,
) -> dict[str, dict[str, Any]]:
    price_by_code = erp_client.fetch_price_list_rates(catalog_price_list(config))
    if catalog_use_highest_price(config):
        invoice_rates = erp_client.fetch_latest_invoice_rates()
        compare_base = catalog_price_compare_currency(config)
        merged: dict[str, dict[str, Any]] = {}
        for item in items:
            code = str(item.get("item_code", "")).strip()
            if not code:
                continue
            picked = pick_catalog_price_entry(
                item_price_entry=price_by_code.get(code),
                invoice_entry=invoice_rates.get(code),
                standard_rate=_positive_rate(item.get("standard_rate")),
                fx=fx,
                compare_base=compare_base,
            )
            if picked is not None:
                merged[code] = picked
        return merged
    if not catalog_invoice_price_fallback(config):
        return price_by_code
    missing: set[str] = set()
    for item in items:
        code = str(item.get("item_code", "")).strip()
        if not code or code in price_by_code:
            continue
        if _positive_rate(item.get("standard_rate")) is not None:
            continue
        missing.add(code)
    if not missing:
        return price_by_code
    invoice_rates = erp_client.fetch_latest_invoice_rates(item_codes=missing)
    if not invoice_rates:
        return price_by_code
    merged = dict(price_by_code)
    merged.update(invoice_rates)
    return merged


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


_SYNC_DATE_LINE = re.compile(r"^- Stock/price as of:.*$", re.MULTILINE)


def catalog_content_for_compare(content: str) -> str:
    """Normalize catalog markdown for change detection (ignore volatile sync timestamp)."""
    without_date = _SYNC_DATE_LINE.sub("", content)
    return without_date.strip() + "\n"


def _group_catalog_items_by_path(
    items: list[dict[str, Any]],
    catalog_root: Path,
) -> dict[Path, list[dict[str, Any]]]:
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for item in items:
        code = str(item.get("item_code", "")).strip()
        if not code:
            continue
        path = catalog_root / f"{safe_catalog_filename(code)}.md"
        grouped.setdefault(path, []).append(item)
    return grouped


def sync_catalog_files(
    settings: Settings,
    slug: str,
    items: list[dict[str, Any]],
    stock_totals: dict[str, float],
    *,
    include_stock: bool,
    price_by_code: dict[str, dict[str, Any]] | None = None,
    sync_date: str | None = None,
) -> CatalogFileSyncResult:
    catalog_root = tenant_catalog_dir(settings, slug)
    sync_label = sync_date or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    logs: list[str] = []
    written = 0
    changed_paths: list[Path] = []
    removed_paths: list[Path] = []
    prices = price_by_code or {}
    grouped = _group_catalog_items_by_path(items, catalog_root)
    active_paths = set(grouped.keys())

    for path, path_items in sorted(grouped.items()):
        path_items.sort(key=lambda item: str(item.get("item_code", "")).strip())
        item = path_items[0]
        code = str(item.get("item_code", "")).strip()
        if len(path_items) > 1:
            codes = [str(entry.get("item_code", "")).strip() for entry in path_items]
            logs.append(
                f"note: {path.name} shared by {len(path_items)} item codes "
                f"({', '.join(codes)}), using {code}"
            )
        stock_qty = stock_totals.get(code) if include_stock else None
        content = render_item_markdown(
            item,
            stock_qty=stock_qty,
            sync_date=sync_label,
            include_stock=include_stock,
            price_entry=prices.get(code),
        )
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if catalog_content_for_compare(existing) == catalog_content_for_compare(content):
                continue
        path.write_text(content, encoding="utf-8")
        written += 1
        changed_paths.append(path)
        logs.append(f"wrote: {path.name}")

    removed = 0
    for path in catalog_root.glob("*.md"):
        if path not in active_paths:
            path.unlink(missing_ok=True)
            removed += 1
            removed_paths.append(path)
            logs.append(f"removed: {path.name}")

    return CatalogFileSyncResult(
        written=written,
        removed=removed,
        logs=logs,
        changed_paths=changed_paths,
        removed_paths=removed_paths,
    )


def catalog_rag_index_plan(
    session: Session,
    tenant_id: int,
    catalog_root: Path,
) -> CatalogRagIndexPlan:
    if not catalog_root.is_dir():
        return CatalogRagIndexPlan(needs_embed=[], already_indexed=[])

    indexed = {
        row.path: row.content_hash
        for row in session.scalars(
            select(IngestedFileRow).where(IngestedFileRow.tenant_id == tenant_id)
        ).all()
    }
    needs_embed: list[Path] = []
    already_indexed: list[Path] = []
    for path in sorted(catalog_root.glob("*.md")):
        key = str(path.resolve())
        digest = file_content_hash(path)
        if indexed.get(key) == digest:
            already_indexed.append(path)
        else:
            needs_embed.append(path)
    return CatalogRagIndexPlan(needs_embed=needs_embed, already_indexed=already_indexed)


def reconcile_catalog_rag(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    slug: str,
    paths_to_reindex: list[Path] | None = None,
    reconcile_all: bool = False,
    batch_size: int = 100,
    pause_seconds: float = 0.0,
    on_file_done: Callable[[Path, str], None] | None = None,
    commit_each_batch: bool = False,
) -> list[str]:
    merged = settings
    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    tenant = tenant_svc.get_by_slug(slug)
    if tenant is not None:
        merged = merge_tenant_settings(settings, tenant)

    catalog_root = tenant_catalog_dir(settings, slug)
    store = LanceVectorStore(settings.lancedb_root / slug)
    embedder = metered_embedder(
        inner=GeminiEmbedder(),
        tenant_id=tenant_id,
        operation="embed_catalog",
        model=merged.embedding_model,
        session=session,
    )
    sync_svc = IngestSyncService(
        settings=merged,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant_id,
    )
    logs = sync_svc.prune_missing_under_root(catalog_root)
    if reconcile_all:
        plan = catalog_rag_index_plan(session, tenant_id, catalog_root)
        md_files = sorted(catalog_root.glob("*.md"))
        logs.append(
            f"index plan: {len(plan.needs_embed)} to embed, "
            f"{len(plan.already_indexed)} already indexed"
        )
    elif paths_to_reindex:
        md_files = sorted({path.resolve() for path in paths_to_reindex if path.is_file()})
    else:
        md_files = []
        logs.append("skipped RAG ingest (no catalog file changes)")

    batch_commit = commit_each_batch or len(md_files) > 50
    if md_files:
        logs.extend(
            sync_svc.ingest_paths_batched(
                md_files,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
                commit_each_batch=batch_commit,
                on_file_done=on_file_done,
            )
        )
    logs.extend(sync_svc.maybe_optimize())
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
    force_rag_reconcile: bool = False,
    skip_rag_ingest: bool = False,
    on_file_done: Callable[[Path, str], None] | None = None,
    commit_each_batch: bool = False,
) -> CatalogSyncResult:
    erp_client = client or ErpNextClient(config)
    include_stock = catalog_include_stock(config)
    sync_date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    try:
        items = erp_client.list_catalog_items()
        stock_totals = erp_client.fetch_stock_totals() if include_stock else {}
        fx = FxRateService(settings.data_root) if catalog_use_highest_price(config) else None
        price_by_code = build_catalog_price_map(
            items,
            erp_client=erp_client,
            config=config,
            fx=fx,
        )
    except Exception as exc:
        return CatalogSyncResult(ok=False, message=str(exc))

    file_result = sync_catalog_files(
        settings,
        tenant_slug,
        items,
        stock_totals,
        include_stock=include_stock,
        price_by_code=price_by_code,
        sync_date=sync_date,
    )
    paths_to_reindex = file_result.changed_paths
    rag_logs: list[str] = []
    if not skip_rag_ingest:
        try:
            rag_logs = reconcile_catalog_rag(
                session,
                settings=settings,
                tenant_id=tenant_id,
                slug=tenant_slug,
                paths_to_reindex=paths_to_reindex,
                reconcile_all=force_rag_reconcile,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
                on_file_done=on_file_done,
                commit_each_batch=commit_each_batch,
            )
        except Exception as exc:
            return CatalogSyncResult(
                ok=False,
                message=str(exc),
                item_count=len(items),
                files_written=file_result.written,
                files_removed=file_result.removed,
                logs=file_result.logs,
            )

    rag_indexed = sum(1 for line in rag_logs if line.startswith("ingested "))
    message = f"Catalog synced: {len(items)} items"
    if skip_rag_ingest:
        message += " (RAG ingest skipped)"
    elif file_result.written == 0 and file_result.removed == 0 and not force_rag_reconcile:
        message += " (no file changes, RAG ingest skipped)"
    elif rag_indexed:
        message += f" ({rag_indexed} files re-indexed)"

    return CatalogSyncResult(
        ok=True,
        message=message,
        item_count=len(items),
        files_written=file_result.written,
        files_removed=file_result.removed,
        rag_files_indexed=rag_indexed,
        logs=file_result.logs + rag_logs,
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


def clear_catalog_sync_metadata(session: Session, integration_id: int) -> None:
    repo = SqlAlchemyIntegrationRepository(session)
    integration = repo.find_by_id(integration_id)
    if integration is None:
        return
    merged = dict(integration.config)
    for key in ("catalog_last_sync_at", "catalog_last_item_count", "catalog_last_error"):
        merged.pop(key, None)
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


def catalog_rag_effective_enabled(*, active: bool, config: dict[str, Any]) -> bool:
    return active and catalog_sync_enabled(config)


def _purge_catalog_rag_vectors(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    tenant_slug: str,
) -> list[str]:
    catalog_dir = tenant_catalog_dir(settings, tenant_slug)
    vector_store = LanceVectorStore(settings.lancedb_root / tenant_slug)
    vector_store.delete_by_source_path_prefix(f"{catalog_dir.resolve()}/")
    rows = list(
        session.scalars(
            select(IngestedFileRow).where(IngestedFileRow.tenant_id == tenant_id)
        ).all()
    )
    under = [row for row in rows if _is_path_under_root(row.path, catalog_dir)]
    for row in under:
        session.delete(row)
    session.flush()
    if under:
        return [f"purged index: {len(under)} paths under {catalog_dir}"]
    return ["no ingested paths under catalog"]


def purge_catalog_files_and_rag(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    tenant_slug: str,
) -> list[str]:
    catalog_dir = tenant_catalog_dir(settings, tenant_slug)
    logs: list[str] = []
    removed = 0
    if catalog_dir.is_dir():
        for path in catalog_dir.glob("*.md"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                logs.append(f"failed to delete: {path}")
    logs.append(f"removed {removed} catalog markdown files")
    logs.extend(
        _purge_catalog_rag_vectors(
            session,
            settings=settings,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
        )
    )
    return logs


def apply_catalog_rag_transition(
    session: Session,
    settings: Settings,
    *,
    tenant_id: int,
    tenant_slug: str,
    integration_id: int,
    config: dict[str, Any],
    prev_enabled: bool,
    now_enabled: bool,
    run_sync_background: Callable[..., None],
) -> None:
    if prev_enabled == now_enabled:
        return
    if now_enabled:
        run_sync_background(
            settings,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
            integration_id=integration_id,
            config=dict(config),
        )
        return
    _purge_catalog_rag_vectors(
        session,
        settings=settings,
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
    )
