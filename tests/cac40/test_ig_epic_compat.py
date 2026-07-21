from chatbot.cac40.config import Cac40Config
from chatbot.cac40.ig_connector import IgConnector


def _ig() -> IgConnector:
    return IgConnector(Cac40Config(), dry_run=True)


def test_epic_product_hint_daily_is_spreadbet() -> None:
    ig = _ig()
    assert ig.epic_product_hint("IX.D.CAC.DAILY.IP") == "SPREADBET"


def test_epic_product_hint_ifs_is_cfd() -> None:
    ig = _ig()
    assert ig.epic_product_hint("IX.D.CAC.IFS.IP") == "CFD"
    assert ig.epic_product_hint("IX.D.CAC.CFS.IP") == "CFD"


def test_cfd_account_rejects_daily_epic() -> None:
    ig = _ig()
    assert ig.epic_compatible_with_account(epic="IX.D.CAC.DAILY.IP", account_type="CFD") is False
    assert ig.epic_compatible_with_account(epic="IX.D.CAC.IFS.IP", account_type="CFD") is True


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
