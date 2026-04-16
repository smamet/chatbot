from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import IngestedFileRow
from chatbot.adapters.rag.chunker import chunk_text
from chatbot.adapters.rag.parsers.registry import parse_file, supported_suffixes
from chatbot.config.settings import Settings
from chatbot.domain.contracts.embedder import Embedder
from chatbot.domain.contracts.vector_store import VectorRecord, VectorStore


class IngestService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        session: Session,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = vector_store
        self._session = session

    def ingest_path(self, path: Path) -> list[str]:
        path = path.resolve()
        logs: list[str] = []
        if path.is_file():
            logs.extend(self._ingest_file(path))
        elif path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.is_file() and p.suffix.lower() in supported_suffixes():
                    logs.extend(self._ingest_file(p))
        else:
            logs.append(f"skip not found: {path}")
        return logs

    def _file_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def _ingest_file(self, file_path: Path) -> list[str]:
        suffix = file_path.suffix.lower()
        if suffix not in supported_suffixes():
            return [f"skip unsupported: {file_path}"]
        key = str(file_path)
        digest = self._file_hash(file_path)
        existing = self._session.scalar(select(IngestedFileRow).where(IngestedFileRow.path == key))
        if existing and existing.content_hash == digest:
            return [f"unchanged: {file_path}"]
        text = parse_file(file_path)
        chunks = chunk_text(
            text,
            source_path=key,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            return [f"no text extracted: {file_path}"]
        self._store.delete_by_source_path(key)
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_texts(texts)
        if len(vectors) != len(chunks):
            return [f"embedding count mismatch: {file_path}"]
        records = [
            VectorRecord(
                chunk_id=c.chunk_id,
                text=c.text,
                source_path=c.source_path,
                vector=vectors[i],
            )
            for i, c in enumerate(chunks)
        ]
        self._store.upsert(records)
        if existing:
            existing.content_hash = digest
        else:
            self._session.add(IngestedFileRow(path=key, content_hash=digest))
        self._session.flush()
        return [f"ingested {len(chunks)} chunks: {file_path}"]
