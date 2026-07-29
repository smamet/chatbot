from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DecisionCache:
    """Persist / replay LLM decisions keyed by bar timestamp."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, bar_ts: str) -> dict[str, Any] | None:
        return self._data.get(bar_ts)

    def put(self, bar_ts: str, decision: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        self._data[bar_ts] = {"decision": decision, "meta": meta or {}}
        self.path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")

    def keys(self) -> list[str]:
        return sorted(self._data.keys())
