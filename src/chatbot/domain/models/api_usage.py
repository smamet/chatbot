from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

ApiUsageOperation = Literal[
    "chat",
    "rewrite",
    "embed_chat",
    "embed_ingest",
    "embed_catalog",
    "trader",
]


@dataclass(frozen=True, slots=True)
class ApiUsageSummary:
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    call_count: int


@dataclass(frozen=True, slots=True)
class ApiUsageDayEntry:
    usage_date: date
    operation: str
    model: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    call_count: int


@dataclass(frozen=True, slots=True)
class TokenDayPoint:
    usage_date: date
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class DiskDayPoint:
    snapshot_date: date
    total_bytes: int
