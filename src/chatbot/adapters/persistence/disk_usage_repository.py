from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import DiskUsageDailyRow
from chatbot.domain.models.api_usage import DiskDayPoint


class SqlAlchemyDiskUsageRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def has_snapshot_for_date(self, snapshot_date: date) -> bool:
        row = self._session.scalar(
            select(func.count())
            .select_from(DiskUsageDailyRow)
            .where(DiskUsageDailyRow.snapshot_date == snapshot_date)
        )
        return int(row or 0) > 0

    def upsert_snapshot(
        self,
        *,
        tenant_id: int | None,
        snapshot_date: date,
        total_bytes: int,
    ) -> None:
        query = select(DiskUsageDailyRow).where(DiskUsageDailyRow.snapshot_date == snapshot_date)
        if tenant_id is None:
            query = query.where(DiskUsageDailyRow.tenant_id.is_(None))
        else:
            query = query.where(DiskUsageDailyRow.tenant_id == tenant_id)
        row = self._session.scalar(query)
        if row is None:
            self._session.add(
                DiskUsageDailyRow(
                    tenant_id=tenant_id,
                    snapshot_date=snapshot_date,
                    total_bytes=total_bytes,
                )
            )
            return
        row.total_bytes = total_bytes

    def tenant_series_since(
        self,
        tenant_id: int,
        since: date,
        until: date | None = None,
    ) -> list[DiskDayPoint]:
        rows = self._session.scalars(
            select(DiskUsageDailyRow)
            .where(
                DiskUsageDailyRow.tenant_id == tenant_id,
                *_snapshot_date_filters(since, until),
            )
            .order_by(DiskUsageDailyRow.snapshot_date)
        ).all()
        return [
            DiskDayPoint(snapshot_date=row.snapshot_date, total_bytes=int(row.total_bytes))
            for row in rows
        ]

    def platform_tenant_sum_series_since(
        self,
        since: date,
        until: date | None = None,
    ) -> list[DiskDayPoint]:
        rows = self._session.execute(
            select(
                DiskUsageDailyRow.snapshot_date,
                func.coalesce(func.sum(DiskUsageDailyRow.total_bytes), 0),
            )
            .where(
                DiskUsageDailyRow.tenant_id.is_not(None),
                *_snapshot_date_filters(since, until),
            )
            .group_by(DiskUsageDailyRow.snapshot_date)
            .order_by(DiskUsageDailyRow.snapshot_date)
        ).all()
        return [
            DiskDayPoint(snapshot_date=row[0], total_bytes=int(row[1]))
            for row in rows
        ]

    def host_series_since(
        self,
        since: date,
        until: date | None = None,
    ) -> list[DiskDayPoint]:
        rows = self._session.scalars(
            select(DiskUsageDailyRow)
            .where(
                DiskUsageDailyRow.tenant_id.is_(None),
                *_snapshot_date_filters(since, until),
            )
            .order_by(DiskUsageDailyRow.snapshot_date)
        ).all()
        return [
            DiskDayPoint(snapshot_date=row.snapshot_date, total_bytes=int(row.total_bytes))
            for row in rows
        ]


def _snapshot_date_filters(since: date, until: date | None) -> tuple:
    filters = [DiskUsageDailyRow.snapshot_date >= since]
    if until is not None:
        filters.append(DiskUsageDailyRow.snapshot_date <= until)
    return tuple(filters)


def today_utc() -> date:
    return datetime.now(UTC).date()
