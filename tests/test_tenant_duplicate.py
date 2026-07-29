"""Tests for bot duplicate (credentials + settings clone)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.application.tenant_duplicate_service import (
    TenantDuplicateError,
    duplicate_tenant,
)
from chatbot.application.tenant_service import TenantService
from chatbot.application.trader_live_service import load_live_config, save_live_config
from chatbot.application.user_service import UserService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.tenant import BotType, TenantConfig, TraderSettings
from chatbot.domain.models.user import UserRole


@pytest.fixture
def dup_ctx(test_settings):
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    yield test_settings, factory, session
    session.close()
    engine.dispose()


def test_duplicate_assistant_copies_connectors_not_docs(dup_ctx) -> None:
    settings, _factory, session = dup_ctx
    tenants = TenantService(SqlAlchemyTenantRepository(session))
    source = tenants.create_tenant(
        name="Source Assistant",
        slug="source-asst",
        prompt="Hello world",
        hook_instructions="Do hooks",
        gemini_api_key="tenant-key-xyz",
        config=TenantConfig(
            rag_top_k=9,
            allowed_connectors=("whatsapp:in",),
            allowed_integrations=("erpnext",),
        ),
        bot_type=BotType.ASSISTANT,
    )
    tenants.update_tenant(source.tenant.id, active=True)
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    conn_svc.upsert(
        tenant_id=source.tenant.id,
        direction=ConnectorDirection.IN,
        type=ConnectorType.WHATSAPP,
        mode=ConnectorMode.DIRECT,
        config={"verify_token": "secret-vt", "access_token": "secret-at"},
        active=True,
    )
    docs = tenant_docs_dir(settings, source.tenant.slug)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "guide.md").write_text("do not copy", encoding="utf-8")

    users = UserService(SqlAlchemyUserRepository(session))
    op = users.create_user(
        email="op@example.com", password="password123", role=UserRole.CLIENT_OPERATOR
    )
    users.grant_access(op.id, source.tenant.id)
    session.commit()

    result = duplicate_tenant(
        session,
        settings,
        source.tenant.slug,
        name="Clone Assistant",
        slug="clone-asst",
    )
    session.commit()

    assert result.tenant.slug == "clone-asst"
    assert result.tenant.id != source.tenant.id
    assert result.token
    assert result.token != source.token
    assert result.tenant.prompt == "Hello world"
    assert result.tenant.hook_instructions == "Do hooks"
    assert result.tenant.gemini_api_key == "tenant-key-xyz"
    assert result.tenant.config.rag_top_k == 9
    assert result.tenant.bot_type == BotType.ASSISTANT

    cloned_conns = conn_svc.list_for_tenant(result.tenant.id)
    assert len(cloned_conns) == 1
    assert cloned_conns[0].config["verify_token"] == "secret-vt"
    assert cloned_conns[0].config["access_token"] == "secret-at"

    clone_docs = tenant_docs_dir(settings, result.tenant.slug)
    assert not clone_docs.exists() or not any(clone_docs.rglob("*"))

    assert result.tenant.id in users.tenant_ids_for_user(op.id)


def test_duplicate_trader_remaps_ig_and_live_config(dup_ctx) -> None:
    settings, _factory, session = dup_ctx
    tenants = TenantService(SqlAlchemyTenantRepository(session))
    source = tenants.create_tenant(
        name="EURUSD Trader",
        slug="eurusd-src",
        prompt="FX prompt",
        bot_type=BotType.TRADER,
        config=TenantConfig(
            trader=TraderSettings(
                market_profile="eurusd",
                symbol="EURUSD",
                epic="CS.D.EURUSD.MINI.IP",
                fundmanager_token="fm-secret",
            )
        ),
    )
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    ig_a = conn_svc.create_ig(
        tenant_id=source.tenant.id,
        config={"api_key": "ig-key-a", "username": "demo1", "password": "pw1"},
        active=True,
    )
    ig_b = conn_svc.create_ig(
        tenant_id=source.tenant.id,
        config={"api_key": "ig-key-b", "username": "demo2", "password": "pw2"},
        active=True,
    )
    save_live_config(
        settings,
        source.tenant.slug,
        {
            "mode": "live",
            "ig_connector_ids": [ig_a.id, ig_b.id],
            "strategy": {
                "order_size": 1.5,
                "max_open_positions": 2,
                "llm_trigger_mode": "interval",
            },
        },
    )
    # Source OHLC must not be copied
    ohlc = Path(settings.data_root) / "trader" / source.tenant.slug / "ohlc"
    ohlc.mkdir(parents=True, exist_ok=True)
    (ohlc / "15m.csv").write_text("ts,open\n", encoding="utf-8")
    session.commit()

    result = duplicate_tenant(
        session,
        settings,
        source.tenant.slug,
        name="CAC Clone",
        slug="cac-clone",
        market_profile="cac40",
        symbol="CAC40",
        epic="IX.D.CAC.BMU.IP",
        reset_prompt_from_profile=False,
    )
    session.commit()

    assert result.tenant.bot_type == BotType.TRADER
    assert result.tenant.config.trader.market_profile == "cac40"
    assert result.tenant.config.trader.symbol == "CAC40"
    assert result.tenant.config.trader.epic == "IX.D.CAC.BMU.IP"
    assert result.tenant.config.trader.fundmanager_token == "fm-secret"
    assert result.tenant.prompt == "FX prompt"

    cloned_igs = conn_svc.list_ig(result.tenant.id)
    assert len(cloned_igs) == 2
    by_user = {c.config.get("username"): c for c in cloned_igs}
    assert by_user["demo1"].config["api_key"] == "ig-key-a"
    assert by_user["demo2"].config["password"] == "pw2"
    assert by_user["demo1"].id != ig_a.id

    live = load_live_config(settings, result.tenant.slug)
    assert live["mode"] == "off"
    assert live["strategy"]["order_size"] == 1.5
    assert set(live["ig_connector_ids"]) == {by_user["demo1"].id, by_user["demo2"].id}

    clone_ohlc = Path(settings.data_root) / "trader" / result.tenant.slug / "ohlc"
    assert not clone_ohlc.exists()


def test_duplicate_missing_source_raises(dup_ctx) -> None:
    settings, _factory, session = dup_ctx
    with pytest.raises(TenantDuplicateError, match="not found"):
        duplicate_tenant(session, settings, "missing-bot", name="Nope")
