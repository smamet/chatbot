from unittest.mock import MagicMock, patch

from chatbot.application.connector_test_service import run_ig_cross_market_working_order_probe


def test_cross_market_probe_blocks_live() -> None:
    result = run_ig_cross_market_working_order_probe(
        {
            "api_key": "key",
            "username": "user",
            "password": "pass",
            "acc_type": "LIVE",
            "epic": "IX.D.CAC.BMU.IP",
        }
    )
    assert result.ok is False
    assert result.error == "live_blocked"


def test_cross_market_probe_requires_credentials() -> None:
    result = run_ig_cross_market_working_order_probe({"acc_type": "DEMO"})
    assert result.ok is False
    assert result.error == "missing_credentials"


def test_cross_market_probe_accepts_forex_shapes() -> None:
    """France 40 shapes fail; Forex LIMIT ACCEPTED → ok."""

    class _FakeIg:
        def __init__(self, *a, **k):
            self._cst = "cst"
            self.config = MagicMock(epic="IX.D.CAC.BMU.IP")
            self._market_cache: dict = {}
            self.cancelled: list[str] = []

        def login(self):
            return None

        def close(self):
            return None

        def get_active_account(self):
            return {"accountType": "CFD", "currency": "USD"}

        def epic_product_hint(self, epic=None, *, market=None):
            return "CFD"

        def list_open_positions(self, epic=""):
            return []

        def list_working_orders(self):
            return []

        def get_market(self, epic=None):
            ep = (epic or self.config.epic or "").upper()
            if "EURUSD" in ep or "EUR/USD" in ep:
                bid, offer, name = 1.10, 1.1002, "EUR/USD Mini"
            elif "TSLA" in ep:
                bid, offer, name = 250.0, 250.2, "Tesla Inc"
            elif "USCGC" in ep or "GOLD" in ep:
                bid, offer, name = 2400.0, 2400.5, "Spot Gold"
            else:
                bid, offer, name = 8400.0, 8402.0, "France 40"
            return {
                "snapshot": {"bid": bid, "offer": offer, "marketStatus": "TRADEABLE"},
                "instrument": {
                    "name": name,
                    "expiry": "-",
                    "currencies": [{"code": "USD"}, {"code": "EUR"}],
                },
                "dealingRules": {
                    "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 2.0},
                    "minDealSize": {"unit": "POINTS", "value": 0.1},
                },
            }

        def search_markets(self, term):
            t = (term or "").lower()
            if "eur" in t:
                return [
                    {
                        "epic": "CS.D.EURUSD.MINI.IP",
                        "instrumentName": "EUR/USD Mini",
                        "marketStatus": "TRADEABLE",
                    }
                ]
            if "tesla" in t:
                return [
                    {
                        "epic": "UA.D.TSLA.CASH.IP",
                        "instrumentName": "Tesla Inc",
                        "marketStatus": "TRADEABLE",
                    }
                ]
            if "gold" in t:
                return [
                    {
                        "epic": "CS.D.USCGC.CFD.IP",
                        "instrumentName": "Spot Gold",
                        "marketStatus": "TRADEABLE",
                    }
                ]
            return []

        def submit_working_order_raw(self, body, version="2"):
            epic = str(body.get("epic") or "")
            if "CAC" in epic:
                return {
                    "dealStatus": "REJECTED",
                    "reason": "ATTACHED_ORDER_LEVEL_ERROR",
                    "dealId": "D1",
                    "dealReference": "R1",
                }
            return {
                "dealStatus": "ACCEPTED",
                "reason": "SUCCESS",
                "dealId": "D2",
                "dealReference": "R2",
            }

        def cancel_working_order(self, deal_id):
            self.cancelled.append(deal_id)
            return {}

    with patch("chatbot.trader.ig_connector.IgConnector", _FakeIg):
        result = run_ig_cross_market_working_order_probe(
            {
                "api_key": "key",
                "username": "user",
                "password": "pass",
                "acc_type": "DEMO",
                "epic": "IX.D.CAC.BMU.IP",
                "account_id": "Z6CXW3",
            }
        )
    assert result.ok is True
    assert "ACCEPTED" in result.message
    assert "EUR/USD" in result.message or "EURUSD" in result.message
    assert "BUY LIMIT" in result.message
    assert "BUY STOP" in result.message
    assert "+TP" in result.message
