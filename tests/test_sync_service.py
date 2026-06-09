from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import Mock

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

    def clear_all(self) -> None:
        self.cleared = True

    def upsert(self, records: list[VectorRecord]) -> None:
        pass

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        return []


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


def test_reconcile_fresh_clears_tenant_and_ingests(sync_session) -> None:
    test_settings, session, store, embedder, tenant = sync_session
    root = _workspace_root(test_settings)

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    svc._ingest.ingest_path = Mock(return_value=["ingest-stub"])
    logs = svc.reconcile_root(root, fresh=True)
    assert store.cleared
    assert "cleared vector index" in logs
    assert "ingest-stub" in logs
    ingest_mock = svc._ingest.ingest_path
    assert isinstance(ingest_mock, Mock)
    ingest_mock.assert_called_once()


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
