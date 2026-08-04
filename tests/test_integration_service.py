from __future__ import annotations

from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from evenor.application.integration_service import IntegrationService
from evenor.domain.models.integration import IntegrationType


def test_integration_upsert_and_find(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyIntegrationRepository(session)
        svc = IntegrationService(repo)
        created = svc.upsert(
            tenant_id=tenant.id,
            type=IntegrationType.ERPNEXT,
            config={"url": "https://erp.test", "api_key": "k", "api_secret": "s"},
            active=True,
        )
        session.commit()
        found = svc.find_active(tenant.id, type=IntegrationType.ERPNEXT)
        assert found is not None
        assert found.config["url"] == "https://erp.test"
        updated = svc.upsert(
            tenant_id=tenant.id,
            type=IntegrationType.ERPNEXT,
            config={"url": "https://erp2.test", "api_key": "k2", "api_secret": "s2"},
            active=True,
        )
        session.commit()
        assert updated.id == created.id
        assert svc.find(tenant.id, type=IntegrationType.ERPNEXT).config["url"] == "https://erp2.test"
    finally:
        session.close()
        engine.dispose()
