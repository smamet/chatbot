from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import IngestedFileRow
from chatbot.application.ingest_service import IngestService
from chatbot.config.settings import Settings
from chatbot.domain.contracts.embedder import Embedder
from chatbot.domain.contracts.vector_store import VectorStore


def _is_path_under_root(path_str: str, root: Path) -> bool:
    """True if stored absolute path is the root file or lies under root directory."""
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
    ) -> None:
        self._session = session
        self._store = vector_store
        self._ingest = IngestService(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            session=session,
        )

    def prune_missing_under_root(self, root: Path) -> list[str]:
        root = root.resolve()
        logs: list[str] = []
        rows = list(self._session.scalars(select(IngestedFileRow)).all())
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

    def reconcile_root(self, root: Path) -> list[str]:
        """Prune missing ingested files under root, then run full ingest for that path."""
        logs: list[str] = []
        logs.extend(self.prune_missing_under_root(root))
        logs.extend(self._ingest.ingest_path(root))
        return logs
