from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import select

from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.orm import IngestedFileRow
from chatbot.application.sync_service import IngestSyncService
from chatbot.domain.contracts.vector_store import RetrievedChunk, VectorRecord
from tests.conftest import TestSettings as SettingsForTests


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.deleted_paths: list[str] = []
        self.cleared = False

    def delete_by_source_path(self, source_path: str) -> None:
        self.deleted_paths.append(source_path)

    def delete_by_source_path_prefix(self, prefix: str) -> None:
        self.deleted_paths.append(prefix)

    def clear_all(self) -> None:
        self.cleared = True

    def upsert(self, records: list[VectorRecord]) -> None:
        pass

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        return []

    def optimize(self, *, cleanup_older_than=None) -> str | None:
        return None


@pytest.fixture
def sync_session(test_settings: SettingsForTests, test_tenant):
    tenant, _ = test_tenant
    test_settings.lancedb_root.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    store = _FakeVectorStore()
    embedder = _FakeEmbedder()
    try:
        with factory() as session:
            yield test_settings, session, store, embedder, tenant
            session.rollback()
    finally:
        engine.dispose()


def _workspace_root(test_settings: SettingsForTests) -> Path:
    return Path(test_settings.database_url.removeprefix("sqlite:///")).parent


def test_prune_calls_vector_delete_and_removes_row(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    missing_key = str((root / "missing.pdf").resolve())
    session.add(IngestedFileRow(tenant_id=tenant.id, path=missing_key, content_hash="dead"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    logs = svc.prune_missing_under_root(root)
    assert missing_key in store.deleted_paths
    assert any("pruned missing" in line for line in logs)
    assert session.scalar(
        select(IngestedFileRow).where(
            IngestedFileRow.tenant_id == tenant.id,
            IngestedFileRow.path == missing_key,
        )
    ) is None


def test_no_prior_when_no_rows_under_root(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    outside = str((root.parent / f"outside_root_{uuid.uuid4().hex}.pdf").resolve())
    session.add(IngestedFileRow(tenant_id=tenant.id, path=outside, content_hash="a"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    logs = svc.prune_missing_under_root(root)
    assert "no prior ingested paths under root" in logs
    assert outside not in store.deleted_paths


def test_clear_tenant_index_removes_tenant_rows_only(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    inside = str((root / "a.pdf").resolve())
    session.add(IngestedFileRow(tenant_id=tenant.id, path=inside, content_hash="a"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    logs = svc.clear_tenant_index()
    assert store.cleared
    assert "cleared vector index" in logs
    assert "cleared 1 ingested file records" in logs
    assert list(
        session.scalars(
            select(IngestedFileRow).where(IngestedFileRow.tenant_id == tenant.id)
        ).all()
    ) == []


def test_reconcile_fresh_purges_root_only_and_ingests(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    docs_root = root / "docs"
    catalog_root = root / "catalog"
    docs_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)
    docs_path = str((docs_root / "a.pdf").resolve())
    catalog_path = str((catalog_root / "item.md").resolve())
    session.add(IngestedFileRow(tenant_id=tenant.id, path=docs_path, content_hash="a"))
    session.add(IngestedFileRow(tenant_id=tenant.id, path=catalog_path, content_hash="b"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(return_value=["ingest-stub"])
    logs = svc.reconcile_root(docs_root, fresh=True)
    assert not store.cleared
    assert any(docs_path in p or p == f"{docs_root.resolve()}/" for p in store.deleted_paths)
    assert catalog_path not in store.deleted_paths
    assert any("purged index" in line for line in logs)
    assert "ingest-stub" in logs
    assert session.scalar(
        select(IngestedFileRow).where(
            IngestedFileRow.tenant_id == tenant.id,
            IngestedFileRow.path == catalog_path,
        )
    ) is not None


def test_reconcile_calls_optimize_when_enabled(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    store.optimize = Mock(return_value="optimized LanceDB table (stats)")  # type: ignore[method-assign]

    svc = IngestSyncService(
        settings=test_settings.model_copy(update={"lancedb_optimize_after_sync": True}),
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(return_value=[])
    logs = svc.reconcile_root(root)
    store.optimize.assert_called_once()  # type: ignore[attr-defined]
    assert any("optimized LanceDB table" in line for line in logs)


def test_reconcile_skips_optimize_when_disabled(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    store.optimize = Mock(return_value="optimized")  # type: ignore[method-assign]

    svc = IngestSyncService(
        settings=test_settings.model_copy(update={"lancedb_optimize_after_sync": False}),
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(return_value=[])
    svc.reconcile_root(root)
    store.optimize.assert_not_called()  # type: ignore[attr-defined]


def test_reconcile_calls_ingest_after_prune(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    missing_key = str((root / "gone.pdf").resolve())
    session.add(IngestedFileRow(tenant_id=tenant.id, path=missing_key, content_hash="b"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(return_value=["ingest-stub"])
    logs = svc.reconcile_root(root)
    assert any("pruned missing" in line for line in logs)
    assert "ingest-stub" in logs
    ingest_mock = svc._ingest.ingest_path
    assert isinstance(ingest_mock, Mock)
    ingest_mock.assert_called_once()


def test_ingest_paths_batched_flushes_in_chunks(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    files = [root / f"doc-{i}.md" for i in range(3)]
    for path in files:
        path.write_text(f"# {path.name}\n", encoding="utf-8")

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(return_value=["ingested"])
    logs = svc.ingest_paths_batched(files, batch_size=2)
    assert logs == ["ingested", "ingested", "ingested"]
    assert svc._ingest.ingest_path.call_count == 3


def test_ingest_paths_batched_on_file_done_and_commit(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)
    files = [root / f"doc-{i}.md" for i in range(3)]
    for path in files:
        path.write_text(f"# {path.name}\n", encoding="utf-8")

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(side_effect=lambda path: [f"ingested: {path.name}"])
    done: list[tuple[Path, str]] = []
    commits: list[bool] = []
    original_commit = session.commit

    def track_commit() -> None:
        commits.append(True)
        original_commit()

    session.commit = track_commit  # type: ignore[method-assign]

    logs = svc.ingest_paths_batched(
        files,
        batch_size=2,
        commit_each_batch=True,
        on_file_done=lambda path, line: done.append((path, line)),
    )
    assert logs == [
        "ingested: doc-0.md",
        "ingested: doc-1.md",
        "ingested: doc-2.md",
    ]
    assert len(done) == 3
    assert done[0][1] == "ingested: doc-0.md"
    assert len(commits) == 2


def test_purge_under_root_removes_vectors_keeps_files(
    test_settings, test_tenant, monkeypatch
) -> None:
    tenant, _token = test_tenant
    catalog_dir = test_settings.data_root / "catalog" / tenant.slug
    catalog_dir.mkdir(parents=True, exist_ok=True)
    md_path = catalog_dir / "item.md"
    md_path.write_text("# item", encoding="utf-8")

    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    deleted_paths: list[str] = []

    class FakeStore:
        def delete_by_source_path(self, path: str) -> None:
            deleted_paths.append(path)

        def delete_by_source_path_prefix(self, prefix: str) -> None:
            deleted_paths.append(prefix)

        def optimize(self, *, cleanup_older_than=None) -> str | None:
            return None

    with factory() as session:
        from chatbot.adapters.persistence.orm import IngestedFileRow

        session.add(
            IngestedFileRow(
                tenant_id=tenant.id,
                path=str(md_path),
                content_hash="abc",
            )
        )
        session.commit()

        from chatbot.application.sync_service import IngestSyncService

        svc = IngestSyncService(
            settings=test_settings,
            embedder=MagicMock(),
            vector_store=FakeStore(),
            session=session,
            tenant_id=tenant.id,
        )
        logs = svc.purge_under_root(catalog_dir)
        session.commit()

    assert md_path.is_file()
    assert any(str(md_path) in p or p.endswith("/") for p in deleted_paths)
    assert any("purged index" in line for line in logs)
