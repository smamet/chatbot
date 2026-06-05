from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import IngestedFileRow
from chatbot.application.ingest_service import IngestService
from chatbot.config.settings import Settings
from chatbot.domain.contracts.embedder import Embedder
from chatbot.domain.contracts.vector_store import VectorStore


def _is_path_under_root(path_str: str, root: Path) -> bool:
    try:
        p = Path(path_str)
        r = root.resolve()
        if r.is_dir():
            return p.is_relative_to(r)
        return p.resolve() == r
    except (OSError, ValueError, RuntimeError):
        return False


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

    def reconcile_root(self, root: Path, *, fresh: bool = False) -> list[str]:
        logs: list[str] = []
        if fresh:
            logs.extend(self.clear_tenant_index())
            logs.extend(self._ingest.ingest_path(root))
            return logs
        logs.extend(self.prune_missing_under_root(root))
        logs.extend(self._ingest.ingest_path(root))
        return logs
