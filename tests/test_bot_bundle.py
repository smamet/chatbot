from __future__ import annotations

import io
import json
import zipfile

import pytest

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.bot_bundle_service import (
    BLACKLIST_NAME,
    ImportMode,
    build_export,
    import_bundle,
)
from chatbot.application.connector_service import ConnectorService
from chatbot.application.tenant_service import TenantService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.tenant import TenantConfig


@pytest.fixture
def bundle_ctx(test_settings):
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    result = tenant_svc.create_tenant(
        name="Export Bot",
        slug="export-bot",
        prompt="Custom prompt",
        hook_instructions="Hook me",
        gemini_api_key="tenant-gemini-key",
        config=TenantConfig(
            rag_enabled=True,
            rag_top_k=7,
            allowed_connectors=("whatsapp:in", "email:out"),
            allowed_integrations=("erpnext",),
        ),
    )
    tenant_svc.update_tenant(result.tenant.id, active=True)
    tenant_svc.add_blocked_sender(result.tenant.id, "spammer@evil.com")
    tenant_svc.add_blocked_sender(result.tenant.id, "noise@test.com")
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    conn_svc.upsert(
        tenant_id=result.tenant.id,
        direction=ConnectorDirection.IN,
        type=ConnectorType.WHATSAPP,
        mode=ConnectorMode.DIRECT,
        config={"verify_token": "secret-vt", "access_token": "secret-at"},
        active=True,
    )
    session.commit()

    docs_dir = tenant_docs_dir(test_settings, result.tenant.slug)
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "guide.md").write_text("doc content", encoding="utf-8")
    (docs_dir / "nested").mkdir(parents=True, exist_ok=True)
    (docs_dir / "nested" / "info.txt").write_text("nested", encoding="utf-8")

    yield test_settings, factory, result.tenant, result.token

    session.close()
    engine.dispose()


def test_build_export_contains_manifest_and_documents(bundle_ctx) -> None:
    settings, factory, tenant, _ = bundle_ctx
    with factory() as session:
        data = build_export(tenant, settings, session)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert "manifest.json" in zf.namelist()
        assert "documents/guide.md" in zf.namelist()
        assert "documents/nested/info.txt" in zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["export_version"] == 1
        assert manifest["source_slug"] == "export-bot"
        assert manifest["bot"]["prompt"] == "Custom prompt"
        assert manifest["bot"]["hook_instructions"] == "Hook me"
        assert manifest["bot"]["gemini_api_key"] == "tenant-gemini-key"
        assert manifest["bot"]["config"]["rag_enabled"] is True
        assert manifest["bot"]["config"]["rag_top_k"] == 7
        assert manifest["bot"]["config"]["allowed_connectors"] == [
            "whatsapp:in",
            "email:out",
        ]
        assert manifest["bot"]["config"]["allowed_integrations"] == [
            "erpnext",
        ]
        assert manifest["email_blocked_senders"] == ["noise@test.com", "spammer@evil.com"]
        assert BLACKLIST_NAME in zf.namelist()
        assert zf.read(BLACKLIST_NAME).decode("utf-8") == "noise@test.com\nspammer@evil.com\n"
        assert len(manifest["connectors"]) == 1
        assert manifest["connectors"][0]["config"]["verify_token"] == "secret-vt"
        assert zf.read("documents/guide.md") == b"doc content"


def test_import_create_round_trip(bundle_ctx) -> None:
    settings, factory, tenant, _ = bundle_ctx
    with factory() as session:
        zip_bytes = build_export(tenant, settings, session)
        result = import_bundle(
            zip_bytes,
            mode=ImportMode.CREATE,
            settings=settings,
            session=session,
            tenant_service=TenantService(SqlAlchemyTenantRepository(session)),
            new_name="Imported Copy",
        )
        session.commit()
        imported = result.tenant

    assert result.token
    assert imported.slug != tenant.slug
    assert imported.name == "Imported Copy"
    assert imported.prompt == "Custom prompt"
    assert imported.hook_instructions == "Hook me"
    assert imported.gemini_api_key == "tenant-gemini-key"
    assert imported.config.rag_top_k == 7
    assert imported.config.allowed_connectors == ("whatsapp:in", "email:out")
    assert imported.config.allowed_integrations == ("erpnext",)
    assert imported.config.email_blocked_senders == ("noise@test.com", "spammer@evil.com")

    docs_dir = tenant_docs_dir(settings, imported.slug)
    assert (docs_dir / "guide.md").read_text(encoding="utf-8") == "doc content"
    assert (docs_dir / "nested" / "info.txt").read_text(encoding="utf-8") == "nested"

    with factory() as session:
        connectors = ConnectorService(SqlAlchemyConnectorRepository(session)).list_for_tenant(
            imported.id
        )
    assert len(connectors) == 1
    assert connectors[0].config["verify_token"] == "secret-vt"


def test_import_overwrite_replaces_docs_and_config(bundle_ctx) -> None:
    settings, factory, tenant, _ = bundle_ctx
    target_slug = tenant.slug
    docs_dir = tenant_docs_dir(settings, target_slug)
    (docs_dir / "old.txt").write_text("old", encoding="utf-8")

    with factory() as session:
        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        tenant_svc.update_tenant(tenant.id, prompt="Before overwrite", name="Old Name")
        zip_bytes = build_export(tenant, settings, session)
        manifest = json.loads(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("manifest.json"))
        manifest["bot"]["prompt"] = "After overwrite"
        manifest["bot"]["name"] = "New Name"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("documents/replaced.md", "fresh doc")
        zip_bytes = buf.getvalue()

        result = import_bundle(
            zip_bytes,
            mode=ImportMode.OVERWRITE,
            settings=settings,
            session=session,
            tenant_service=tenant_svc,
            target_slug=target_slug,
        )
        session.commit()

    assert result.token is None
    assert result.tenant.slug == target_slug
    assert result.tenant.prompt == "After overwrite"
    assert result.tenant.name == "New Name"
    assert not (docs_dir / "old.txt").exists()
    assert (docs_dir / "replaced.md").read_text(encoding="utf-8") == "fresh doc"
