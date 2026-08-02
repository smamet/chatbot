"""Disk cache for IG ``GET /markets?searchTerm=`` (24h TTL).

Market search is a normal REST call — not LLM tokens and not the historical
``/prices`` allowance. Caching avoids re-hitting IG on every autocomplete keystroke.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 24 * 60 * 60


def _cache_key(acc_type: str, term: str) -> str:
    raw = f"{(acc_type or 'DEMO').strip().upper()}|{term.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def cache_dir_for(data_root: Path) -> Path:
    return Path(data_root) / "trader" / ".ig-market-search-cache"


def read_cached_search(
    cache_dir: Path,
    *,
    acc_type: str,
    term: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> list[dict[str, Any]] | None:
    path = cache_dir / f"{_cache_key(acc_type, term)}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        cached_at = float(payload.get("cached_at") or 0)
    except (TypeError, ValueError):
        return None
    if cached_at <= 0 or (time.time() - cached_at) > max(60, int(ttl_seconds)):
        return None
    rows = payload.get("markets")
    if not isinstance(rows, list):
        return None
    return [r for r in rows if isinstance(r, dict)]


def write_cached_search(
    cache_dir: Path,
    *,
    acc_type: str,
    term: str,
    markets: list[dict[str, Any]],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(acc_type, term)}.json"
    slim: list[dict[str, Any]] = []
    for row in markets:
        if not isinstance(row, dict):
            continue
        slim.append(
            {
                "epic": row.get("epic"),
                "instrumentName": row.get("instrumentName") or row.get("instrument"),
                "marketStatus": row.get("marketStatus"),
                "expiry": row.get("expiry"),
                "instrumentType": row.get("instrumentType"),
            }
        )
    path.write_text(
        json.dumps(
            {
                "cached_at": time.time(),
                "acc_type": (acc_type or "DEMO").strip().upper(),
                "term": term,
                "markets": slim,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def cached_search_markets(
    ig: Any,
    term: str,
    *,
    cache_dir: Path | None,
    acc_type: str = "",
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Return ``(rows, from_cache)``.

    One IG HTTP call per unique (acc_type, term) within ``ttl_seconds``.
    """
    q = (term or "").strip()
    if not q:
        return [], True
    acc = (acc_type or getattr(getattr(ig, "config", None), "ig_acc_type", None) or "DEMO")
    acc = str(acc).strip().upper() or "DEMO"
    if cache_dir is not None:
        hit = read_cached_search(cache_dir, acc_type=acc, term=q, ttl_seconds=ttl_seconds)
        if hit is not None:
            return hit, True
    rows = list(ig.search_markets(q) or [])
    rows = [r for r in rows if isinstance(r, dict)]
    if cache_dir is not None:
        try:
            write_cached_search(cache_dir, acc_type=acc, term=q, markets=rows)
        except Exception:
            logger.debug("Failed to write IG market search cache", exc_info=True)
    return rows, False
