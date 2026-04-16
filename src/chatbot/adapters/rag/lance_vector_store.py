from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np

from chatbot.domain.contracts.vector_store import RetrievedChunk, VectorRecord, VectorStore


class LanceVectorStore:
    _TABLE = "chunks"

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))

    def delete_by_source_path(self, source_path: str) -> None:
        if self._TABLE not in self._db.table_names():
            return
        tbl = self._db.open_table(self._TABLE)
        safe = source_path.replace("'", "''")
        tbl.delete(f"source_path == '{safe}'")

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        rows = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "source_path": r.source_path,
                "vector": np.asarray(r.vector, dtype=np.float32),
            }
            for r in records
        ]
        if self._TABLE not in self._db.table_names():
            self._db.create_table(self._TABLE, data=rows)
        else:
            self._db.open_table(self._TABLE).add(rows)

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        if self._TABLE not in self._db.table_names():
            return []
        tbl = self._db.open_table(self._TABLE)
        if tbl.count_rows() == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        results = tbl.search(q).limit(top_k).to_pandas()
        out: list[RetrievedChunk] = []
        for _, row in results.iterrows():
            raw = row.get("_distance", row.get("_score"))
            score = float(raw) if raw is not None and not (isinstance(raw, float) and np.isnan(raw)) else 0.0
            out.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    source_path=str(row["source_path"]),
                    score=score,
                )
            )
        return out
