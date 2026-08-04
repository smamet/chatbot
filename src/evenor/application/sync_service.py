from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from evenor.adapters.persistence.orm import IngestedFileRow
from evenor.application.ingest_service import IngestService
from evenor.config.settings import Settings
from evenor.domain.contracts.embedder import Embedder
from evenor.domain.contracts.vector_store import VectorStore


def _is_path_under_root(path_str: str, root: Path) -> bool:
    try:
        p = Path(path_str)
        r = root.resolve()
        if r.is_dir():
            return p.is_relative_to(r)
        return p.resolve() == r
    except (OSError, ValueError, RuntimeError):
        return False


def _maybe_optimize_vector_store(store: VectorStore, settings: Settings) -> list[str]:
    if not settings.lancedb_optimize_after_sync:
        return []
    days = max(0, settings.lancedb_cleanup_older_than_days)
    cleanup = timedelta(days=days)
    try:
        msg = store.optimize(cleanup_older_than=cleanup)
        return [msg] if msg else []
    except Exception as exc:
        return [f"LanceDB optimize skipped: {exc}"]


class IngestSyncService:
    """Prune index entries for removed files under a root, then (re)ingest everything there."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        session: Session,
        tenant_id: int,
    ) -> None:
        self._session = session
        self._settings = settings
        self._store = vector_store
        self._tenant_id = tenant_id
        self._ingest = IngestService(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            session=session,
            tenant_id=tenant_id,
        )

    def prune_missing_under_root(self, root: Path) -> list[str]:
        root = root.resolve()
        logs: list[str] = []
        rows = list(
            self._session.scalars(
                select(IngestedFileRow).where(IngestedFileRow.tenant_id == self._tenant_id)
            ).all()
        )
        under = [r for r in rows if _is_path_under_root(r.path, root)]
        if not under:
            logs.append("no prior ingested paths under root")
        for row in under:
            if Path(row.path).is_file():
                continue
            self._store.delete_by_source_path(row.path)
            self._session.delete(row)
            logs.append(f"pruned missing: {row.path}")
        self._session.flush()
        return logs

    def purge_under_root(self, root: Path) -> list[str]:
        """Remove vector rows and ingested_files records under root; keep source files on disk."""
        root = root.resolve()
        logs: list[str] = []
        rows = list(
            self._session.scalars(
                select(IngestedFileRow).where(IngestedFileRow.tenant_id == self._tenant_id)
            ).all()
        )
        under = [r for r in rows if _is_path_under_root(r.path, root)]
        if under:
            self._store.delete_by_source_path_prefix(f"{root}/")
            for row in under:
                self._session.delete(row)
            logs.append(f"purged index: {len(under)} paths under {root}")
        self._session.flush()
        if not under:
            logs.append("no ingested paths under root")
        return logs

    def clear_tenant_index(self) -> list[str]:
        n = (
            self._session.scalar(
                select(func.count())
                .select_from(IngestedFileRow)
                .where(IngestedFileRow.tenant_id == self._tenant_id)
            )
            or 0
        )
        self._store.clear_all()
        self._session.execute(
            delete(IngestedFileRow).where(IngestedFileRow.tenant_id == self._tenant_id)
        )
        self._session.flush()
        return ["cleared vector index", f"cleared {n} ingested file records"]

    def maybe_optimize(self) -> list[str]:
        return _maybe_optimize_vector_store(self._store, self._settings)

    def reconcile_root(self, root: Path, *, fresh: bool = False) -> list[str]:
        logs: list[str] = []
        if fresh:
            logs.extend(self.purge_under_root(root))
        else:
            logs.extend(self.prune_missing_under_root(root))
        logs.extend(self._ingest.ingest_path(root))
        logs.extend(self.maybe_optimize())
        return logs

    def ingest_paths_batched(
        self,
        paths: list[Path],
        *,
        batch_size: int = 100,
        pause_seconds: float = 0.0,
        commit_each_batch: bool = False,
        on_file_done: Callable[[Path, str], None] | None = None,
    ) -> list[str]:
        logs: list[str] = []
        if not paths:
            return logs
        size = max(1, batch_size)
        for index in range(0, len(paths), size):
            batch = paths[index : index + size]
            for path in batch:
                if path.is_file():
                    file_logs = self._ingest.ingest_path(path)
                    logs.extend(file_logs)
                    if on_file_done is not None:
                        on_file_done(path, file_logs[-1] if file_logs else "")
            if commit_each_batch:
                self._session.commit()
            else:
                self._session.flush()
            if pause_seconds > 0 and index + size < len(paths):
                time.sleep(pause_seconds)
        return logs
