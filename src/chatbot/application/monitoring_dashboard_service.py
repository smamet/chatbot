from __future__ import annotations

import json
import math
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository
from chatbot.adapters.persistence.disk_usage_repository import SqlAlchemyDiskUsageRepository
from chatbot.application.disk_usage_service import DiskUsageService
from chatbot.application.monitoring_date_range import MONITORING_PAGE_SIZE
from chatbot.application.monitoring_series import (
    disk_chart_payload,
    disk_pie_chart_payload,
    fill_disk_series,
    fill_token_series,
    token_chart_payload,
)
from chatbot.application.usage_cost_service import CostProfile, UsageCostService
from chatbot.application.usage_recorder_service import UsageRecorderService
from chatbot.config.settings import Settings
from chatbot.domain.models.tenant import Tenant


class MonitoringDashboardService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._usage_repo = SqlAlchemyApiUsageRepository(session)
        self._disk_repo = SqlAlchemyDiskUsageRepository(session)
        self._recorder = UsageRecorderService(self._usage_repo)
        self._cost = UsageCostService()
        self._disk_usage = DiskUsageService(settings)

    def bot_context(
        self,
        tenant: Tenant,
        *,
        since: date,
        until: date,
        is_admin: bool,
        usage_page: int = 1,
    ) -> dict:
        usage_days = (until - since).days + 1
        usage_summary = self._recorder.tenant_summary_since(tenant.id, since, until)
        usage_daily = self._recorder.tenant_daily_since(tenant.id, since, until)
        disk_usage = self._disk_usage.tenant_usage(tenant.slug)

        token_points = fill_token_series(
            self._recorder.tenant_token_series_since(tenant.id, since, until),
            since,
            until,
        )
        disk_points = fill_disk_series(
            self._disk_repo.tenant_series_since(tenant.id, since, until),
            since,
            until,
        )

        profile: CostProfile = "internal" if is_admin else "client"
        cost = self._cost.estimate_cost(
            usage_daily,
            profile=profile,
            settings=self._settings,
            tenant=tenant,
        )

        usage_row_total = self._recorder.tenant_daily_count(tenant.id, since, until)
        usage_page_count = max(1, math.ceil(usage_row_total / MONITORING_PAGE_SIZE))
        page = min(max(1, usage_page), usage_page_count)
        offset = (page - 1) * MONITORING_PAGE_SIZE
        page_entries = self._recorder.tenant_daily_page(
            tenant.id,
            since,
            until,
            offset=offset,
            limit=MONITORING_PAGE_SIZE,
        )
        usage_rows = self._cost.row_costs(
            page_entries,
            profile=profile,
            settings=self._settings,
            tenant=tenant,
        )
        row_start = offset + 1 if usage_row_total else 0
        row_end = min(offset + len(page_entries), usage_row_total)

        ctx: dict = {
            "usage_days": usage_days,
            "usage_from": since,
            "usage_to": until,
            "usage_summary": usage_summary,
            "usage_daily": usage_daily,
            "usage_rows": usage_rows,
            "usage_page": page,
            "usage_page_count": usage_page_count,
            "usage_row_total": usage_row_total,
            "usage_page_size": MONITORING_PAGE_SIZE,
            "usage_row_start": row_start,
            "usage_row_end": row_end,
            "disk_usage": disk_usage,
            "cost_estimate": cost,
            "cost_profile": profile,
            "usage_chart_json": json.dumps(token_chart_payload(token_points)),
            "disk_chart_json": json.dumps(disk_chart_payload(disk_points, label="Bot disk")),
        }
        if is_admin:
            ctx["client_cost_estimate"] = self._cost.estimate_cost(
                usage_daily,
                profile="client",
                settings=self._settings,
                tenant=tenant,
            )
        return ctx

    def global_context(
        self,
        tenants: list[Tenant],
        *,
        since: date,
        until: date,
    ) -> dict:
        usage_days = (until - since).days + 1
        usage_by_tenant = self._recorder.all_tenant_summaries_since(since, until)
        daily_by_tenant = self._recorder.all_tenant_daily_since(since, until)
        disk_svc = self._disk_usage

        token_points = fill_token_series(
            self._recorder.platform_token_series_since(since, until),
            since,
            until,
        )
        platform_disk = fill_disk_series(
            self._disk_repo.platform_tenant_sum_series_since(since, until),
            since,
            until,
        )
        host_disk = fill_disk_series(
            self._disk_repo.host_series_since(since, until),
            since,
            until,
        )

        host_live = disk_svc.host_usage()

        rows = []
        cost_total = Decimal("0")
        for tenant in tenants:
            usage = usage_by_tenant.get(
                tenant.id,
                self._recorder.tenant_summary_since(tenant.id, since, until),
            )
            daily = daily_by_tenant.get(tenant.id, [])
            internal = self._cost.estimate_cost(
                daily,
                profile="internal",
                settings=self._settings,
                tenant=tenant,
            )
            client = self._cost.estimate_cost(
                daily,
                profile="client",
                settings=self._settings,
                tenant=tenant,
            )
            cost_total += internal.total_usd
            rows.append(
                {
                    "tenant": tenant,
                    "disk": disk_svc.tenant_usage(tenant.slug),
                    "usage": usage,
                    "internal_cost": internal,
                    "client_cost": client,
                }
            )

        return {
            "usage_days": usage_days,
            "usage_from": since,
            "usage_to": until,
            "rows": rows,
            "host_disk": host_live,
            "platform_internal_cost_usd": cost_total,
            "usage_chart_json": json.dumps(token_chart_payload(token_points)),
            "disk_bot_chart_json": json.dumps(
                disk_chart_payload(platform_disk, label="Bot data (all bots)")
            ),
            "disk_host_chart_json": json.dumps(
                disk_chart_payload(host_disk, label="Host volume (DATA_ROOT)")
            ),
            "disk_pie_chart_json": json.dumps(
                disk_pie_chart_payload(
                    used_bytes=host_live.used_bytes,
                    free_bytes=host_live.free_bytes,
                )
            ),
        }
