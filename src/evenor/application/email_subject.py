from __future__ import annotations

import re

from rapidfuzz import fuzz

_SUBJECT_PREFIX = re.compile(
    r"^(?:re|fwd?|fw|tr|aw|r|reply)\s*:\s*",
    re.IGNORECASE,
)
_BRACKET_TAG = re.compile(r"^\[[^\]]+\]\s*")
_EXTERNAL_TAG = re.compile(r"^\[external\]\s*", re.IGNORECASE)


def normalize_subject(subject: str | None) -> str:
    text = (subject or "").strip()
    if not text:
        return ""
    changed = True
    while changed:
        changed = False
        for pattern in (_EXTERNAL_TAG, _BRACKET_TAG, _SUBJECT_PREFIX):
            m = pattern.match(text)
            if m:
                text = text[m.end() :].strip()
                changed = True
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def subject_similarity(a: str, b: str) -> float:
    left = normalize_subject(a)
    right = normalize_subject(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return fuzz.token_sort_ratio(left, right) / 100.0
