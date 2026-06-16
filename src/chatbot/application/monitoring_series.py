from __future__ import annotations

from datetime import date, timedelta
from typing import TypeVar

from chatbot.domain.models.api_usage import DiskDayPoint, TokenDayPoint

T = TypeVar("T")


def date_range_inclusive(since: date, until: date) -> list[date]:
    if until < since:
        return []
    days: list[date] = []
    current = since
    while current <= until:
        days.append(current)
        current += timedelta(days=1)
    return days


def fill_token_series(points: list[TokenDayPoint], since: date, until: date) -> list[TokenDayPoint]:
    by_date = {p.usage_date: p for p in points}
    return [
        by_date.get(
            d,
            TokenDayPoint(usage_date=d, prompt_tokens=0, output_tokens=0),
        )
        for d in date_range_inclusive(since, until)
    ]


def fill_disk_series(points: list[DiskDayPoint], since: date, until: date) -> list[DiskDayPoint]:
    by_date = {p.snapshot_date: p for p in points}
    return [
        by_date.get(
            d,
            DiskDayPoint(snapshot_date=d, total_bytes=0),
        )
        for d in date_range_inclusive(since, until)
    ]


def token_chart_payload(points: list[TokenDayPoint]) -> dict:
    return {
        "labels": [p.usage_date.isoformat() for p in points],
        "prompt_tokens": [p.prompt_tokens for p in points],
        "output_tokens": [p.output_tokens for p in points],
    }


def disk_chart_payload(points: list[DiskDayPoint], *, label: str) -> dict:
    return {
        "label": label,
        "labels": [p.snapshot_date.isoformat() for p in points],
        "total_bytes": [p.total_bytes for p in points],
    }
