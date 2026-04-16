from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_path: str
    score: float


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk_id: str
    text: str
    source_path: str
    vector: list[float]


@runtime_checkable
class VectorStore(Protocol):
    def delete_by_source_path(self, source_path: str) -> None:
        """Remove all chunks for a logical source (e.g. before re-ingesting the same file)."""
        ...

    def upsert(self, records: list[VectorRecord]) -> None:
        ...

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        ...
