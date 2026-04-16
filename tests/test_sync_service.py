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

    def delete_by_source_path(self, source_path: str) -> None:
        self.deleted_paths.append(source_path)

    def upsert(self, records: list[VectorRecord]) -> None:
        pass

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        return []


@pytest.fixture
def sync_session(test_settings: SettingsForTests):
    test_settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(test_settings)
    factory = session_factory(engine)
    store = _FakeVectorStore()
    embedder = _FakeEmbedder()
    try:
        with factory() as session:
            yield test_settings, session, store, embedder
            session.rollback()
    finally:
        engine.dispose()


def _workspace_root(test_settings: SettingsForTests) -> Path:
    return Path(test_settings.database_url.removeprefix("sqlite:///")).parent


def test_prune_calls_vector_delete_and_removes_row(sync_session) -> None:
    test_settings, session, store, embedder = sync_session
    root = _workspace_root(test_settings)
    missing_key = str((root / "missing.pdf").resolve())
    session.add(IngestedFileRow(path=missing_key, content_hash="dead"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
    )
    logs = svc.prune_missing_under_root(root)
    assert missing_key in store.deleted_paths
    assert any("pruned missing" in line for line in logs)
    assert session.scalar(select(IngestedFileRow).where(IngestedFileRow.path == missing_key)) is None


def test_no_prior_when_no_rows_under_root(sync_session) -> None:
    test_settings, session, store, embedder = sync_session
    root = _workspace_root(test_settings)
    outside = str((root.parent / f"outside_root_{uuid.uuid4().hex}.pdf").resolve())
    session.add(IngestedFileRow(path=outside, content_hash="a"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
    )
    logs = svc.prune_missing_under_root(root)
    assert "no prior ingested paths under root" in logs
    assert outside not in store.deleted_paths


def test_reconcile_calls_ingest_after_prune(sync_session) -> None:
    test_settings, session, store, embedder = sync_session
    root = _workspace_root(test_settings)
    missing_key = str((root / "gone.pdf").resolve())
    session.add(IngestedFileRow(path=missing_key, content_hash="b"))
    session.flush()

    svc = IngestSyncService(
        settings=test_settings,
        embedder=embedder,
        vector_store=store,
        session=session,
    )
    svc._ingest.ingest_path = Mock(return_value=["ingest-stub"])
    logs = svc.reconcile_root(root)
    assert any("pruned missing" in line for line in logs)
    assert "ingest-stub" in logs
    ingest_mock = svc._ingest.ingest_path
    assert isinstance(ingest_mock, Mock)
    ingest_mock.assert_called_once()
