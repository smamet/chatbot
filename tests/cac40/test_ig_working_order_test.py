from unittest.mock import MagicMock, patch

from chatbot.application.connector_test_service import run_ig_working_order_test
from chatbot.cac40.models import OrderType, Side


def test_ig_working_order_test_blocks_live() -> None:
    result = run_ig_working_order_test(
        {
            "api_key": "key",
            "username": "user",
            "password": "pass",
            "acc_type": "LIVE",
            "epic": "IX.D.CAC.DAILY.IP",
        }
    )
    assert result.ok is False
    assert result.error == "live_blocked"
    assert "DEMO-only" in result.message


def test_ig_working_order_test_requires_credentials() -> None:
    result = run_ig_working_order_test({"acc_type": "DEMO"})
    assert result.ok is False
    assert result.error == "missing_credentials"


def test_ig_working_order_test_places_limit_with_attached_tp() -> None:
    """DEMO test must pass limit_level (take-profit) on LIMIT entries."""
    placed_kwargs: list[dict] = []

    cancelled: set[str] = set()

    class _FakeIg:
        def __init__(self, *a, **k):
            self._cst = "cst"
            self.config = MagicMock(epic="IX.D.CAC.IFD.IP")

        def login(self):
            return None

        def close(self):
            return None

        def get_active_account(self):
            return {"accountType": "CFD", "currency": "EUR"}

        def epic_product_hint(self, epic=None):
            return "CFD"

        def epic_compatible_with_account(self, **k):
            return True

        def sync_price(self):
            return 8450.0

        def resolve_order_currency(self):
            return "EUR"

        def resolve_min_deal_size(self):
            return 1.0

        def resolve_order_expiry(self):
            return "-"

        def market_currency_codes(self):
            return ["EUR"]

        def get_market(self):
            return {"snapshot": {"marketStatus": "TRADEABLE"}}

        def snap_level(self, level):
            return float(level)

        def place_order(self, order, *, currency=None, limit_level=None, stop_level=None):
            placed_kwargs.append(
                {
                    "side": order.side,
                    "type": order.type,
                    "level": order.level,
                    "limit_level": limit_level,
                }
            )
            order.deal_id = f"D{len(placed_kwargs)}"
            order.client_ref = f"R{len(placed_kwargs)}"
            return order

        def list_working_orders(self):
            out = []
            for i, kw in enumerate(placed_kwargs, start=1):
                did = f"D{i}"
                if did in cancelled:
                    continue
                row = {
                    "dealId": did,
                    "direction": kw["side"].value,
                    "orderType": kw["type"].value,
                    "orderLevel": kw["level"],
                    "epic": "IX.D.CAC.IFD.IP",
                }
                if kw["limit_level"] is not None:
                    row["limitLevel"] = kw["limit_level"]
                out.append(row)
            return out

        def cancel_working_order(self, deal_id):
            cancelled.add(str(deal_id))
            return {"dealId": deal_id}

    with (
        patch(
            "chatbot.cac40.ig_connector.IgConnector",
            _FakeIg,
        ),
        patch("time.sleep", return_value=None),
    ):
        result = run_ig_working_order_test(
            {
                "api_key": "key",
                "username": "user",
                "password": "pass",
                "acc_type": "DEMO",
                "epic": "IX.D.CAC.IFD.IP",
                "order_size": 1.0,
            },
            hold_seconds=0.1,
        )

    assert result.ok is True, result.message
    limit_places = [p for p in placed_kwargs if p["type"] == OrderType.LIMIT]
    assert len(limit_places) == 3  # bare BUY, BUY+TP, SELL+TP
    with_tp = [p for p in limit_places if p["limit_level"] is not None]
    assert len(with_tp) == 2
    # BUY entry below mid → TP above entry (and above mid); SELL → TP below
    buy = next(p for p in with_tp if p["side"] == Side.BUY)
    sell = next(p for p in with_tp if p["side"] == Side.SELL)
    assert buy["limit_level"] > buy["level"]
    assert sell["limit_level"] < sell["level"]
    assert "TP@" in result.message
    assert "attached" in result.message.lower() or "limitLevel" in result.message
