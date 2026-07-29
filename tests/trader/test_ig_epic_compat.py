from unittest.mock import MagicMock

from chatbot.trader.config import TraderConfig
from chatbot.trader.ig_connector import IgConnector


def _ig() -> IgConnector:
    return IgConnector(TraderConfig(), dry_run=True)


def test_cancel_working_order_uses_post_method_override() -> None:
    """IG rejects bare DELETE; must POST with _method=DELETE (trading-ig pattern)."""
    ig = IgConnector(TraderConfig(epic="IX.D.CAC.BMU.IP"), dry_run=False)
    ig._cst = "cst"
    ig._security = "sec"
    resp = MagicMock()
    resp.is_error = False
    resp.content = b"{}"
    resp.json.return_value = {"dealReference": "x"}
    ig._client = MagicMock()
    ig._client.post.return_value = resp
    ig.cancel_working_order("DIAAAAX5S7QLQA8")
    ig._client.post.assert_called_once()
    ig._client.delete.assert_not_called()
    headers = ig._client.post.call_args.kwargs.get("headers") or {}
    assert headers.get("_method") == "DELETE"
    assert ig._client.post.call_args.kwargs.get("json") == {}


def test_epic_product_hint_daily_without_currency_is_unknown() -> None:
    """IX.D.CAC.DAILY.IP is IG's France 40 Cash CFD — not auto spread-bet."""
    ig = _ig()
    ig.market_currency_codes = lambda epic=None: []  # type: ignore[method-assign]
    assert ig.epic_product_hint("IX.D.CAC.DAILY.IP") == "UNKNOWN"


def test_epic_product_hint_daily_eur_is_cfd() -> None:
    ig = _ig()
    ig.market_currency_codes = lambda epic=None: ["EUR"]  # type: ignore[method-assign]
    assert ig.epic_product_hint("IX.D.CAC.DAILY.IP") == "CFD"


def test_epic_product_hint_gbp_only_is_spreadbet() -> None:
    ig = _ig()
    ig.market_currency_codes = lambda epic=None: ["GBP"]  # type: ignore[method-assign]
    assert ig.epic_product_hint("IX.D.FTSE.DAILY.IP") == "SPREADBET"


def test_epic_product_hint_ifs_is_cfd() -> None:
    ig = _ig()
    assert ig.epic_product_hint("IX.D.CAC.IFS.IP") == "CFD"
    assert ig.epic_product_hint("IX.D.CAC.CFS.IP") == "CFD"


def test_cfd_account_accepts_daily_cac_when_unknown_or_cfd() -> None:
    ig = _ig()
    ig.market_currency_codes = lambda epic=None: []  # type: ignore[method-assign]
    assert ig.epic_compatible_with_account(epic="IX.D.CAC.DAILY.IP", account_type="CFD") is True
    ig.market_currency_codes = lambda epic=None: ["EUR"]  # type: ignore[method-assign]
    assert ig.epic_compatible_with_account(epic="IX.D.CAC.DAILY.IP", account_type="CFD") is True
    assert ig.epic_compatible_with_account(epic="IX.D.CAC.IFS.IP", account_type="CFD") is True


def test_cfd_account_rejects_gbp_only_epic() -> None:
    ig = _ig()
    ig.market_currency_codes = lambda epic=None: ["GBP"]  # type: ignore[method-assign]
    assert ig.epic_compatible_with_account(epic="IX.D.FTSE.DAILY.IP", account_type="CFD") is False


def test_snap_level_removes_float_noise() -> None:
    ig = _ig()
    ig._market_cache = {
        "IX.D.CAC.DAILY.IP": {
            "dealingRules": {"minStepDistance": {"unit": "POINTS", "value": 0.1}},
        }
    }
    ig.config.epic = "IX.D.CAC.DAILY.IP"
    assert ig.snap_level(8293.900000000001) == 8293.9


def test_resolve_order_expiry_keeps_undated_for_cfd() -> None:
    ig = _ig()
    ig.config.epic = "IX.D.CAC.IFS.IP"
    ig._market_cache = {
        "IX.D.CAC.IFS.IP": {"instrument": {"expiry": "-"}},
    }
    assert ig.resolve_order_expiry() == "-"
