from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.disk_usage_repository import SqlAlchemyDiskUsageRepository, today_utc
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.disk_snapshot_service import DiskSnapshotService
from chatbot.application.disk_usage_service import DiskUsageService
from chatbot.application.tenant_service import TenantService
from chatbot.config.settings import reset_settings_cache_for_tests


@pytest.fixture
def snapshot_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Session:
    db = tmp_path / "snap.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    reset_settings_cache_for_tests()
    from chatbot.config.settings import get_settings

    engine = create_db_engine(get_settings(), for_tests=True)
    factory = session_factory(engine)
    session = factory()
    TenantService(SqlAlchemyTenantRepository(session)).create_tenant(name="Snap Bot", slug="snap-bot")
    session.commit()
    yield session
    session.close()


def test_disk_snapshot_idempotent(snapshot_session: Session, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chatbot.config.settings import get_settings

    settings = get_settings()
    svc = DiskSnapshotService(
        settings=settings,
        disk_repo=SqlAlchemyDiskUsageRepository(snapshot_session),
        tenant_repo=SqlAlchemyTenantRepository(snapshot_session),
        disk_usage=DiskUsageService(settings),
    )
    first = svc.record_all_if_due()
    snapshot_session.commit()
    assert first
    second = svc.record_all_if_due()
    assert second == []
