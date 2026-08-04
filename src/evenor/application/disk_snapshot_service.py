from __future__ import annotations

from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.disk_usage_service import DiskUsageService
from evenor.adapters.persistence.disk_usage_repository import SqlAlchemyDiskUsageRepository, today_utc
from evenor.config.settings import Settings


class DiskSnapshotService:
    def __init__(
        self,
        *,
        settings: Settings,
        disk_repo: SqlAlchemyDiskUsageRepository,
        tenant_repo: SqlAlchemyTenantRepository,
        disk_usage: DiskUsageService,
    ) -> None:
        self._settings = settings
        self._disk_repo = disk_repo
        self._tenant_repo = tenant_repo
        self._disk_usage = disk_usage

    def record_all_if_due(self) -> list[str]:
        if not self._settings.disk_snapshot_enabled:
            return []
        today = today_utc()
        if self._disk_repo.has_snapshot_for_date(today):
            return []
        return self.record_all()

    def record_all(self) -> list[str]:
        today = today_utc()
        tenants = self._tenant_repo.list_all()
        for tenant in tenants:
            usage = self._disk_usage.tenant_usage(tenant.slug)
            self._disk_repo.upsert_snapshot(
                tenant_id=tenant.id,
                snapshot_date=today,
                total_bytes=usage.total_bytes,
            )
        host = self._disk_usage.host_usage()
        self._disk_repo.upsert_snapshot(
            tenant_id=None,
            snapshot_date=today,
            total_bytes=host.used_bytes,
        )
        return [f"disk snapshot: {len(tenants)} tenants + host ({today.isoformat()})"]
