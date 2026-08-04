from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session

from evenor.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.application.usage_recorder_service import UsageRecorderService
from evenor.config.settings import reset_settings_cache_for_tests
from evenor.domain.contracts.llm_client import LlmUsage


@pytest.fixture
def usage_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Session:
    db = tmp_path / "usage.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    reset_settings_cache_for_tests()
    from evenor.config.settings import get_settings

    engine = create_db_engine(get_settings(), for_tests=True)
    factory = session_factory(engine)
    session = factory()
    yield session
    session.close()


def test_usage_recorder_increments_daily_row(usage_session: Session) -> None:
    repo = SqlAlchemyApiUsageRepository(usage_session)
    recorder = UsageRecorderService(repo)
    recorder.record(
        1,
        "chat",
        "gemini-2.0-flash",
        LlmUsage(prompt_tokens=10, candidates_tokens=5, total_tokens=15),
    )
    recorder.record(
        1,
        "chat",
        "gemini-2.0-flash",
        LlmUsage(prompt_tokens=3, candidates_tokens=2, total_tokens=5),
    )
    usage_session.commit()

    since = date(2000, 1, 1)
    summary = recorder.tenant_summary_since(1, since)
    assert summary.prompt_tokens == 13
    assert summary.output_tokens == 7
    assert summary.total_tokens == 20
    assert summary.call_count == 2

    daily = recorder.tenant_daily_since(1, since)
    assert len(daily) == 1
    assert daily[0].operation == "chat"
    assert daily[0].model == "gemini-2.0-flash"


def test_usage_recorder_separates_operations_and_models(usage_session: Session) -> None:
    repo = SqlAlchemyApiUsageRepository(usage_session)
    recorder = UsageRecorderService(repo)
    recorder.record(2, "chat", "gemini-2.0-flash", LlmUsage(prompt_tokens=1, candidates_tokens=1))
    recorder.record(2, "rewrite", "gemini-2.0-flash", LlmUsage(prompt_tokens=2, candidates_tokens=0))
    recorder.record(2, "embed_chat", "text-embedding-004", LlmUsage(prompt_tokens=4, candidates_tokens=0))
    usage_session.commit()

    since = date(2000, 1, 1)
    daily = recorder.tenant_daily_since(2, since)
    assert len(daily) == 3
    assert {row.operation for row in daily} == {"chat", "rewrite", "embed_chat"}


def test_all_tenant_summaries_since(usage_session: Session) -> None:
    repo = SqlAlchemyApiUsageRepository(usage_session)
    recorder = UsageRecorderService(repo)
    recorder.record(10, "chat", "m", LlmUsage(prompt_tokens=5, candidates_tokens=1, total_tokens=6))
    recorder.record(11, "chat", "m", LlmUsage(prompt_tokens=1, candidates_tokens=2, total_tokens=3))
    usage_session.commit()

    summaries = recorder.all_tenant_summaries_since(date(2000, 1, 1))
    assert summaries[10].total_tokens == 6
    assert summaries[11].total_tokens == 3


def test_all_tenant_daily_since(usage_session: Session) -> None:
    repo = SqlAlchemyApiUsageRepository(usage_session)
    recorder = UsageRecorderService(repo)
    recorder.record(10, "chat", "m", LlmUsage(prompt_tokens=5, candidates_tokens=1, total_tokens=6))
    recorder.record(11, "chat", "m", LlmUsage(prompt_tokens=1, candidates_tokens=2, total_tokens=3))
    usage_session.commit()

    daily = recorder.all_tenant_daily_since(date(2000, 1, 1))
    assert len(daily[10]) == 1
    assert daily[10][0].total_tokens == 6
    assert daily[11][0].total_tokens == 3
    assert daily.get(99, []) == []
