from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository, usage_since_date
from chatbot.adapters.persistence.disk_usage_repository import SqlAlchemyDiskUsageRepository, today_utc
from chatbot.application.disk_usage_service import DiskUsageService
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


def _until() -> date:
    return today_utc()


class MonitoringDashboardService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._usage_repo = SqlAlchemyApiUsageRepository(session)
        self._disk_repo = SqlAlchemyDiskUsageRepository(session)
        self._recorder = UsageRecorderService(self._usage_repo)
        self._cost = UsageCostService()
        self._disk_usage = DiskUsageService(settings)

    def bot_context(self, tenant: Tenant, *, days: int, is_admin: bool) -> dict:
        since = usage_since_date(days)
        until = _until()
        usage_summary = self._recorder.tenant_summary_since(tenant.id, since)
        usage_daily = self._recorder.tenant_daily_since(tenant.id, since)
        disk_usage = self._disk_usage.tenant_usage(tenant.slug)

        token_points = fill_token_series(
            self._recorder.tenant_token_series_since(tenant.id, since),
            since,
            until,
        )
        disk_points = fill_disk_series(
            self._disk_repo.tenant_series_since(tenant.id, since),
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
        usage_rows = self._cost.row_costs(
            usage_daily,
            profile=profile,
            settings=self._settings,
            tenant=tenant,
        )

        ctx: dict = {
            "usage_days": days,
            "usage_summary": usage_summary,
            "usage_daily": usage_daily,
            "usage_rows": usage_rows,
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

    def global_context(self, tenants: list[Tenant], *, days: int) -> dict:
        since = usage_since_date(days)
        until = _until()
        usage_by_tenant = self._recorder.all_tenant_summaries_since(since)
        disk_svc = self._disk_usage

        token_points = fill_token_series(
            self._recorder.platform_token_series_since(since),
            since,
            until,
        )
        platform_disk = fill_disk_series(
            self._disk_repo.platform_tenant_sum_series_since(since),
            since,
            until,
        )
        host_disk = fill_disk_series(
            self._disk_repo.host_series_since(since),
            since,
            until,
        )

        host_live = disk_svc.host_usage()

        rows = []
        cost_total = Decimal("0")
        for tenant in tenants:
            usage = usage_by_tenant.get(
                tenant.id,
                self._recorder.tenant_summary_since(tenant.id, since),
            )
            daily = self._recorder.tenant_daily_since(tenant.id, since)
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
            "usage_days": days,
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
