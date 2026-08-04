"""Tenant delete must clear all tenant-scoped tables and filesystem roots."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import func, select

from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.orm import (
    ApiUsageDailyRow,
    ConnectorRow,
    DiskUsageDailyRow,
    IntegrationRow,
    MailConnectionRow,
    MessageRow,
    TenantRow,
)
from evenor.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from evenor.application.tenant_service import TenantService
from evenor.domain.models.tenant import TenantConfig
from tests.conftest import TestSettings


def test_delete_tenant_removes_related_rows_and_dirs(test_settings: TestSettings) -> None:
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        svc = TenantService(SqlAlchemyTenantRepository(session))
        created = svc.create_tenant(
            name="Trader",
            slug="cac-trader",
            prompt="x",
            config=TenantConfig(rag_enabled=False),
        )
        tid = created.tenant.id
        now = datetime.now(UTC)
        session.add_all(
            [
                ConnectorRow(
                    tenant_id=tid,
                    direction="both",
                    type="ig",
                    mode="direct",
                    config_enc="",
                    active=True,
                ),
                IntegrationRow(
                    tenant_id=tid,
                    type="cac40_backtest",
                    config_enc="",
                    active=True,
                ),
                MailConnectionRow(
                    tenant_id=tid,
                    label="box",
                    provider="microsoft",
                    mailbox_email="a@b.c",
                    config_enc="",
                    active=True,
                ),
                MessageRow(
                    tenant_id=tid,
                    session_id="s1",
                    role="user",
                    content="hi",
                    created_at=now,
                ),
                ApiUsageDailyRow(
                    tenant_id=tid,
                    usage_date=date(2026, 1, 1),
                    operation="cac40",
                    model="gemini",
                    prompt_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    call_count=1,
                ),
                DiskUsageDailyRow(
                    tenant_id=tid,
                    snapshot_date=date(2026, 1, 1),
                    total_bytes=10,
                ),
            ]
        )
        session.commit()

        for rel in ("docs", "catalog", "attachments", "quotes", "cac40"):
            (test_settings.data_root / rel / "cac-trader").mkdir(parents=True, exist_ok=True)
            (test_settings.data_root / rel / "cac-trader" / "marker.txt").write_text("x")
        (test_settings.lancedb_root / "cac-trader").mkdir(parents=True, exist_ok=True)

        assert svc.delete_tenant(tid, settings=test_settings) is True
        session.commit()

        assert session.get(TenantRow, tid) is None
        assert session.scalar(select(func.count()).select_from(ConnectorRow)) == 0
        assert session.scalar(select(func.count()).select_from(IntegrationRow)) == 0
        assert session.scalar(select(func.count()).select_from(MailConnectionRow)) == 0
        assert session.scalar(select(func.count()).select_from(MessageRow)) == 0
        assert session.scalar(select(func.count()).select_from(ApiUsageDailyRow)) == 0
        assert (
            session.scalar(
                select(func.count())
                .select_from(DiskUsageDailyRow)
                .where(DiskUsageDailyRow.tenant_id.is_not(None))
            )
            == 0
        )
        for rel in ("docs", "catalog", "attachments", "quotes", "cac40"):
            assert not (test_settings.data_root / rel / "cac-trader").exists()
        assert not (test_settings.lancedb_root / "cac-trader").exists()
    finally:
        session.close()
