from __future__ import annotations

import sys
import types as pytypes
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.application.usage_recorder_service import UsageRecorderService
from chatbot.trader.llm_decision import GeminiDecisionClient
from chatbot.trader.models import MarketSnapshot
from chatbot.config.settings import reset_settings_cache_for_tests


@pytest.fixture
def usage_factory(tmp_path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "trader_usage.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    reset_settings_cache_for_tests()
    from chatbot.config.settings import get_settings

    engine = create_db_engine(get_settings(), for_tests=True)
    return session_factory(engine)


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch, response: object) -> MagicMock:
    models = MagicMock()
    models.generate_content.return_value = response
    client = MagicMock()
    client.models = models

    google_mod = pytypes.ModuleType("google")
    genai_mod = pytypes.ModuleType("google.genai")
    types_mod = pytypes.ModuleType("google.genai.types")

    class Part:
        @staticmethod
        def from_text(**kwargs):
            return kwargs.get("text")

        @staticmethod
        def from_bytes(**kwargs):
            return kwargs.get("data")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types_mod.Part = Part
    types_mod.GenerateContentConfig = GenerateContentConfig
    genai_mod.Client = MagicMock(return_value=client)
    genai_mod.types = types_mod
    google_mod.genai = genai_mod

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return models


def test_decide_records_trader_usage(usage_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    decision_json = """
    {
      "analysis": {
        "support": 8000,
        "resistance": 8100,
        "bias": "hold",
        "rsi_note": "",
        "pivot_note": ""
      },
      "actions": []
    }
    """
    response = SimpleNamespace(
        text=decision_json,
        candidates=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120,
        ),
    )
    models = _install_fake_genai(monkeypatch, response)

    llm = GeminiDecisionClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        tenant_id=42,
        session_factory=usage_factory,
    )
    result = llm.decide(
        images={"15m": b"\x89PNG"},
        snapshot=MarketSnapshot(symbol="CAC40", last_price=8050.0),
        phase="Flat",
    )
    assert result is not None
    assert result.analysis.bias == "hold"
    models.generate_content.assert_called_once()

    session: Session = usage_factory()
    try:
        recorder = UsageRecorderService(SqlAlchemyApiUsageRepository(session))
        daily = recorder.tenant_daily_since(42, date(2000, 1, 1))
        assert len(daily) == 1
        assert daily[0].operation == "trader"
        assert daily[0].model == "gemini-2.5-flash"
        assert daily[0].prompt_tokens == 100
        assert daily[0].output_tokens == 20
        assert daily[0].total_tokens == 120
        assert daily[0].call_count == 1
    finally:
        session.close()


def test_record_usage_skipped_without_tenant(usage_factory) -> None:
    llm = GeminiDecisionClient(api_key="x", session_factory=usage_factory)
    llm._record_usage(
        SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=1,
                total_token_count=2,
            )
        )
    )
    session = usage_factory()
    try:
        recorder = UsageRecorderService(SqlAlchemyApiUsageRepository(session))
        assert recorder.tenant_daily_since(1, date(2000, 1, 1)) == []
    finally:
        session.close()
