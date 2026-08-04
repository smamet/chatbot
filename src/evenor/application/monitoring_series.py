from __future__ import annotations

from datetime import date, timedelta
from typing import TypeVar

from evenor.domain.models.api_usage import DiskDayPoint, TokenDayPoint

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


def multi_disk_chart_payload(*series: tuple[str, list[DiskDayPoint]]) -> dict:
    if not series:
        return {"labels": [], "series": []}
    expected_len = len(series[0][1])
    if not all(len(points) == expected_len for _, points in series):
        raise ValueError("series length mismatch")
    labels = [p.snapshot_date.isoformat() for p in series[0][1]]
    return {
        "labels": labels,
        "series": [
            {"label": label, "total_bytes": [p.total_bytes for p in points]}
            for label, points in series
        ],
    }


def disk_pie_chart_payload(*, used_bytes: int, free_bytes: int) -> dict:
    return {
        "used_bytes": max(0, int(used_bytes)),
        "free_bytes": max(0, int(free_bytes)),
    }
