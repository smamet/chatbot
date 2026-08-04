"""Mauritian / Morisien marker tokens — whole-token match after letter extraction.

Short markers (e.g. ``la``, ``sa``) can appear in French/English; the rewrite gate uses
only this list (no fastText). Tune ``CREOLE_MARKERS`` for your traffic.

``eki`` is common in Kreol but not in the original marker list; it is included here for practical matching.
"""

from __future__ import annotations

import re

# User-supplied set (deduped). "eki" added — "ki" alone matches whole token only (avoids "ski").
CREOLE_MARKERS: frozenset[str] = frozenset(
    {
        "ki",
        "sa",
        "la",
        "mo",
        "to",
        "li",
        "ti",
        "nou",
        "zot",
        "ena",
        "pa",
        "napa",
        "pou",
        "ek",
        "fer",
        "dir",
        "kouma",
        "bonzour",
        "capav",
        "kifer",
        "kan",
        "kot",
        "kuma",
        "lerla",
        "korek",
        "eki",
    }
)

# Letters (incl. basic Latin accents); avoids digits/punctuation as tokens.
_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def creole_markers_hit(text: str) -> bool:
    """True if any whole token equals a marker (case-insensitive)."""
    return any(t in CREOLE_MARKERS for t in _tokens(text))


def looks_like_mauritian_creole_script(text: str) -> bool:
    """Backward-compatible alias for ``creole_markers_hit``."""
    return creole_markers_hit(text)
