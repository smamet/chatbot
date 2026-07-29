"""Unit tests for Lightstreamer session fields + CAC order matrix (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chatbot.application.connector_test_service import (
    run_ig_cac40_working_order_matrix,
    run_ig_stream_order_probe,
)
from chatbot.trader.ig_stream_probe import lightstreamer_password, run_ig_stream_probe
from chatbot.trader.models import OrderType


def test_lightstreamer_password_format() -> None:
    assert lightstreamer_password("abc", "xyz") == "CST-abc|XST-xyz"


def test_login_captures_lightstreamer_endpoint() -> None:
    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgConnector

    cfg = TraderConfig(
        ig_api_key="k",
        ig_username="u",
        ig_password="p",
        ig_acc_type="DEMO",
        epic="IX.D.CAC.BMU.IP",
    )
    ig = IgConnector(cfg, dry_run=True)

    class _Resp:
        is_error = False
        content = b"{}"
        headers = {"CST": "cst1", "X-SECURITY-TOKEN": "xst1"}

        def json(self):
            return {
                "lightstreamerEndpoint": "https://demo-apd.marketdatasystems.com",
                "currentAccountId": "ABC123",
                "accounts": [{"accountId": "ABC123"}],
            }

    with patch.object(ig._client, "post", return_value=_Resp()):
        ig.login()

    assert ig._cst == "cst1"
    assert ig._security == "xst1"
    assert ig.lightstreamer_endpoint == "https://demo-apd.marketdatasystems.com"
    assert ig.current_account_id == "ABC123"
    ig.close()


def test_stream_probe_blocks_live() -> None:
    result = run_ig_stream_probe(
        {
            "api_key": "k",
            "username": "u",
            "password": "p",
            "acc_type": "LIVE",
        }
    )
    assert result.ok is False
    assert result.error == "live_blocked"


def test_cac40_matrix_places_four_shapes_no_market() -> None:
    placed: list[dict] = []
    market_calls: list[str] = []

    class _FakeIg:
        def __init__(self, *a, **k):
            self._cst = "cst"
            self._security = "xst"
            self.lightstreamer_endpoint = None
            self.current_account_id = "ACC"
            self.config = MagicMock(epic="IX.D.CAC.BMU.IP")

        def login(self):
            return None

        def close(self):
            return None

        def get_active_account(self):
            return {"accountType": "CFD", "currency": "EUR"}

        def epic_compatible_with_account(self, **k):
            return True

        def find_compatible_epic(self, **k):
            return None, []

        def get_market(self, epic=None):
            return {
                "snapshot": {"bid": 8399.0, "offer": 8401.0, "marketStatus": "TRADEABLE"},
                "dealingRules": {},
                "instrument": {"name": "France 40"},
            }

        def resolve_order_currency(self):
            return "EUR"

        def resolve_min_deal_size(self):
            return 1.0

        def resolve_min_stop_or_limit_distance(self, epic=None):
            return 12.0

        def resolve_max_stop_or_limit_distance(self, epic=None):
            return 100.0

        def snap_level(self, level):
            return float(level)

        def place_order(self, order, *, currency=None, limit_level=None, stop_level=None):
            placed.append(
                {
                    "type": order.type,
                    "level": order.level,
                    "limit_level": limit_level,
                }
            )
            order.deal_id = f"D{len(placed)}"
            return order

        def cancel_working_order(self, deal_id):
            return {"dealId": deal_id}

        def open_market_position(self, *a, **k):
            market_calls.append("open")
            return "leg"

        def market_close(self, *a, **k):
            market_calls.append("close")

    with patch("chatbot.trader.ig_connector.IgConnector", _FakeIg):
        result = run_ig_cac40_working_order_matrix(
            {
                "api_key": "k",
                "username": "u",
                "password": "p",
                "acc_type": "DEMO",
                "epic": "IX.D.CAC.BMU.IP",
            },
            allow_market_orders=False,
            use_stream_confirms=False,
        )

    assert result.ok is True, result.message
    assert len(placed) == 4
    assert [p["type"] for p in placed] == [
        OrderType.LIMIT,
        OrderType.LIMIT,
        OrderType.STOP,
        OrderType.STOP,
    ]
    assert placed[0]["limit_level"] is None
    assert placed[1]["limit_level"] is not None
    assert placed[2]["limit_level"] is None
    assert placed[3]["limit_level"] is not None
    assert market_calls == []
    assert "allow_market_orders=false" in result.message


def test_cac40_matrix_market_flag() -> None:
    market_calls: list[str] = []

    class _FakeIg:
        def __init__(self, *a, **k):
            self._cst = "cst"
            self._security = "xst"
            self.lightstreamer_endpoint = None
            self.current_account_id = "ACC"
            self.config = MagicMock(epic="IX.D.CAC.BMU.IP")

        def login(self):
            return None

        def close(self):
            return None

        def get_active_account(self):
            return {"accountType": "CFD", "currency": "EUR"}

        def epic_compatible_with_account(self, **k):
            return True

        def find_compatible_epic(self, **k):
            return None, []

        def get_market(self, epic=None):
            return {
                "snapshot": {"bid": 8399.0, "offer": 8401.0, "marketStatus": "TRADEABLE"},
                "dealingRules": {},
                "instrument": {"name": "France 40"},
            }

        def resolve_order_currency(self):
            return "EUR"

        def resolve_min_deal_size(self):
            return 1.0

        def resolve_min_stop_or_limit_distance(self, epic=None):
            return 12.0

        def resolve_max_stop_or_limit_distance(self, epic=None):
            return 100.0

        def snap_level(self, level):
            return float(level)

        def place_order(self, order, *, currency=None, limit_level=None, stop_level=None):
            order.deal_id = "D1"
            return order

        def cancel_working_order(self, deal_id):
            return {"dealId": deal_id}

        def open_market_position(self, *a, **k):
            market_calls.append("open")
            return "leg1"

        def market_close(self, position_id):
            market_calls.append(f"close:{position_id}")

    with patch("chatbot.trader.ig_connector.IgConnector", _FakeIg):
        result = run_ig_cac40_working_order_matrix(
            {
                "api_key": "k",
                "username": "u",
                "password": "p",
                "acc_type": "DEMO",
            },
            allow_market_orders=True,
            use_stream_confirms=False,
        )

    assert result.ok is True
    assert market_calls == ["open", "close:leg1"]


def test_stream_order_probe_still_runs_fx_when_cac_fails() -> None:
    class _FakeFxIg:
        def __init__(self, *a, **k):
            self._cst = "cst"
            self.config = MagicMock(epic="CS.D.EURUSD.MINI.IP", order_size=1.0)

        def login(self):
            return None

        def close(self):
            return None

        def get_market(self, epic=None):
            return {"snapshot": {"bid": 1.10, "offer": 1.1002}}

        def resolve_order_currency(self):
            return "USD"

        def resolve_min_deal_size(self):
            return 0.1

        def resolve_min_stop_or_limit_distance(self, epic=None):
            return 0.001

        def snap_level(self, level):
            return float(level)

        def place_order(self, order, *, currency=None, limit_level=None, stop_level=None):
            order.deal_id = "FX1"
            return order

        def cancel_working_order(self, deal_id):
            return {"dealId": deal_id}

    with (
        patch(
            "chatbot.application.connector_test_service.run_ig_cac40_working_order_matrix",
            return_value=MagicMock(ok=False, message="cac fail", error="matrix_failed"),
        ),
        patch("chatbot.trader.ig_connector.IgConnector", _FakeFxIg),
    ):
        result = run_ig_stream_order_probe(
            {"api_key": "k", "username": "u", "password": "p", "acc_type": "DEMO"}
        )
    assert result.ok is False  # CAC failed
    assert "cac fail" in result.message
    assert "PASS FX LIMIT bare" in result.message
    assert "PASS FX STOP bare" in result.message


@pytest.mark.integration
def test_live_demo_cac_matrix_and_market() -> None:
    """Live DEMO: CAC LIMIT/STOP ±TP + market open/close. Requires DB IG connector."""
    from sqlalchemy import text

    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.engine import create_db_engine, session_factory
    from chatbot.application.connector_service import ConnectorService
    from chatbot.config.settings import get_settings

    Factory = session_factory(create_db_engine(get_settings()))
    with Factory() as session:
        svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        cfg = None
        for row in session.execute(text("SELECT id FROM tenants")).fetchall():
            cfg = svc.get_ig_config(row[0])
            if cfg:
                break
    if not cfg:
        pytest.skip("No IG connector in DB")
    cfg = dict(cfg)
    cfg["acc_type"] = "DEMO"
    result = run_ig_cac40_working_order_matrix(
        cfg,
        allow_market_orders=True,
        use_stream_confirms=True,
    )
    assert result.ok is True, result.message
