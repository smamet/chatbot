from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import ApiUsageDailyRow
from chatbot.domain.models.api_usage import ApiUsageDayEntry, ApiUsageSummary, TokenDayPoint


class SqlAlchemyApiUsageRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def increment(
        self,
        *,
        tenant_id: int,
        usage_date: date,
        operation: str,
        model: str,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
        call_count: int = 1,
    ) -> None:
        row = self._session.scalar(
            select(ApiUsageDailyRow).where(
                ApiUsageDailyRow.tenant_id == tenant_id,
                ApiUsageDailyRow.usage_date == usage_date,
                ApiUsageDailyRow.operation == operation,
                ApiUsageDailyRow.model == model,
            )
        )
        if row is None:
            self._session.add(
                ApiUsageDailyRow(
                    tenant_id=tenant_id,
                    usage_date=usage_date,
                    operation=operation,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    call_count=call_count,
                )
            )
            return
        row.prompt_tokens += prompt_tokens
        row.output_tokens += output_tokens
        row.total_tokens += total_tokens
        row.call_count += call_count

    def tenant_summary_since(
        self,
        tenant_id: int,
        since: date,
        until: date | None = None,
    ) -> ApiUsageSummary:
        query = select(
            func.coalesce(func.sum(ApiUsageDailyRow.prompt_tokens), 0),
            func.coalesce(func.sum(ApiUsageDailyRow.output_tokens), 0),
            func.coalesce(func.sum(ApiUsageDailyRow.total_tokens), 0),
            func.coalesce(func.sum(ApiUsageDailyRow.call_count), 0),
        ).where(
            ApiUsageDailyRow.tenant_id == tenant_id,
            *_usage_date_filters(since, until),
        )
        row = self._session.execute(query).one()
        return ApiUsageSummary(
            prompt_tokens=int(row[0]),
            output_tokens=int(row[1]),
            total_tokens=int(row[2]),
            call_count=int(row[3]),
        )

    def tenant_daily_since(
        self,
        tenant_id: int,
        since: date,
        until: date | None = None,
    ) -> list[ApiUsageDayEntry]:
        rows = self._session.scalars(
            select(ApiUsageDailyRow)
            .where(
                ApiUsageDailyRow.tenant_id == tenant_id,
                *_usage_date_filters(since, until),
            )
            .order_by(ApiUsageDailyRow.usage_date.desc(), ApiUsageDailyRow.operation)
        ).all()
        return [_row_to_entry(r) for r in rows]

    def tenant_daily_page(
        self,
        tenant_id: int,
        since: date,
        until: date,
        *,
        offset: int,
        limit: int,
    ) -> list[ApiUsageDayEntry]:
        rows = self._session.scalars(
            select(ApiUsageDailyRow)
            .where(
                ApiUsageDailyRow.tenant_id == tenant_id,
                *_usage_date_filters(since, until),
            )
            .order_by(ApiUsageDailyRow.usage_date.desc(), ApiUsageDailyRow.operation)
            .offset(offset)
            .limit(limit)
        ).all()
        return [_row_to_entry(r) for r in rows]

    def tenant_daily_count(
        self,
        tenant_id: int,
        since: date,
        until: date,
    ) -> int:
        row = self._session.scalar(
            select(func.count())
            .select_from(ApiUsageDailyRow)
            .where(
                ApiUsageDailyRow.tenant_id == tenant_id,
                *_usage_date_filters(since, until),
            )
        )
        return int(row or 0)

    def all_tenant_daily_since(
        self,
        since: date,
        until: date | None = None,
    ) -> dict[int, list[ApiUsageDayEntry]]:
        rows = self._session.scalars(
            select(ApiUsageDailyRow)
            .where(*_usage_date_filters(since, until))
            .order_by(
                ApiUsageDailyRow.tenant_id,
                ApiUsageDailyRow.usage_date.desc(),
                ApiUsageDailyRow.operation,
            )
        ).all()
        by_tenant: dict[int, list[ApiUsageDayEntry]] = {}
        for row in rows:
            by_tenant.setdefault(int(row.tenant_id), []).append(_row_to_entry(row))
        return by_tenant

    def all_tenant_summaries_since(
        self,
        since: date,
        until: date | None = None,
    ) -> dict[int, ApiUsageSummary]:
        rows = self._session.execute(
            select(
                ApiUsageDailyRow.tenant_id,
                func.coalesce(func.sum(ApiUsageDailyRow.prompt_tokens), 0),
                func.coalesce(func.sum(ApiUsageDailyRow.output_tokens), 0),
                func.coalesce(func.sum(ApiUsageDailyRow.total_tokens), 0),
                func.coalesce(func.sum(ApiUsageDailyRow.call_count), 0),
            )
            .where(*_usage_date_filters(since, until))
            .group_by(ApiUsageDailyRow.tenant_id)
        ).all()
        return {
            int(tenant_id): ApiUsageSummary(
                prompt_tokens=int(prompt),
                output_tokens=int(output),
                total_tokens=int(total),
                call_count=int(calls),
            )
            for tenant_id, prompt, output, total, calls in rows
        }

    def tenant_token_series_since(
        self,
        tenant_id: int,
        since: date,
        until: date | None = None,
    ) -> list[TokenDayPoint]:
        rows = self._session.execute(
            select(
                ApiUsageDailyRow.usage_date,
                func.coalesce(func.sum(ApiUsageDailyRow.prompt_tokens), 0),
                func.coalesce(func.sum(ApiUsageDailyRow.output_tokens), 0),
            )
            .where(
                ApiUsageDailyRow.tenant_id == tenant_id,
                *_usage_date_filters(since, until),
            )
            .group_by(ApiUsageDailyRow.usage_date)
            .order_by(ApiUsageDailyRow.usage_date)
        ).all()
        return [
            TokenDayPoint(
                usage_date=row[0],
                prompt_tokens=int(row[1]),
                output_tokens=int(row[2]),
            )
            for row in rows
        ]

    def platform_token_series_since(
        self,
        since: date,
        until: date | None = None,
    ) -> list[TokenDayPoint]:
        rows = self._session.execute(
            select(
                ApiUsageDailyRow.usage_date,
                func.coalesce(func.sum(ApiUsageDailyRow.prompt_tokens), 0),
                func.coalesce(func.sum(ApiUsageDailyRow.output_tokens), 0),
            )
            .where(*_usage_date_filters(since, until))
            .group_by(ApiUsageDailyRow.usage_date)
            .order_by(ApiUsageDailyRow.usage_date)
        ).all()
        return [
            TokenDayPoint(
                usage_date=row[0],
                prompt_tokens=int(row[1]),
                output_tokens=int(row[2]),
            )
            for row in rows
        ]


def _usage_date_filters(since: date, until: date | None) -> tuple:
    filters = [ApiUsageDailyRow.usage_date >= since]
    if until is not None:
        filters.append(ApiUsageDailyRow.usage_date <= until)
    return tuple(filters)


def _row_to_entry(row: ApiUsageDailyRow) -> ApiUsageDayEntry:
    return ApiUsageDayEntry(
        usage_date=row.usage_date,
        operation=row.operation,
        model=row.model,
        prompt_tokens=row.prompt_tokens,
        output_tokens=row.output_tokens,
        total_tokens=row.total_tokens,
        call_count=row.call_count,
    )


def usage_since_date(days: int) -> date:
    return datetime.now(UTC).date() - timedelta(days=max(days - 1, 0))
