from __future__ import annotations

from chatbot.trader.epic_resolve import (
    alias_epic_for_symbol,
    looks_like_ig_epic,
    rank_search_rows,
    resolve_ticker_to_epic,
    search_terms_for_symbol,
)


def test_looks_like_ig_epic() -> None:
    assert looks_like_ig_epic("CS.D.EURUSD.MINI.IP")
    assert looks_like_ig_epic("IX.D.CAC.BMU.IP")
    assert not looks_like_ig_epic("EURUSD")
    assert not looks_like_ig_epic("CAC40")


def test_search_terms_for_eurusd() -> None:
    terms = search_terms_for_symbol("EURUSD")
    assert "EURUSD" in terms
    assert "EUR/USD" in terms


def test_alias_epic_for_symbol() -> None:
    assert alias_epic_for_symbol("EURUSD") == "CS.D.EURUSD.MINI.IP"
    assert alias_epic_for_symbol("CAC40") == "IX.D.CAC.BMU.IP"


def test_rank_prefers_mini_cfd() -> None:
    rows = [
        {
            "epic": "CS.D.EURUSD.CFD.IP",
            "instrumentName": "EUR/USD",
            "marketStatus": "TRADEABLE",
        },
        {
            "epic": "CS.D.EURUSD.MINI.IP",
            "instrumentName": "EUR/USD Mini",
            "marketStatus": "TRADEABLE",
        },
        {
            "epic": "CS.D.EURUSD.BULL.IP",
            "instrumentName": "EURUSD Bull",
            "marketStatus": "TRADEABLE",
        },
    ]
    ranked = rank_search_rows(rows, symbol="EURUSD", account_type="CFD")
    assert ranked
    assert ranked[0][1] == "CS.D.EURUSD.MINI.IP"


def test_resolve_offline_alias() -> None:
    resolved = resolve_ticker_to_epic(None, "EURUSD", profile_id="eurusd")
    assert resolved is not None
    assert resolved.epic == "CS.D.EURUSD.MINI.IP"


def test_resolve_explicit_epic() -> None:
    resolved = resolve_ticker_to_epic(
        None, "EURUSD", explicit_epic="CS.D.EURUSD.MINI.IP"
    )
    assert resolved is not None
    assert resolved.source == "explicit"
    assert resolved.epic == "CS.D.EURUSD.MINI.IP"


class _FakeIg:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def search_markets(self, term: str) -> list[dict]:
        return list(self._rows)

    def resolve_account_type(self) -> str:
        return "CFD"

    def epic_product_hint(self, epic: str | None = None) -> str:
        return "CFD"

    def epic_compatible_with_account(self, *, epic=None, account_type=None) -> bool:
        return True


def test_resolve_via_search() -> None:
    ig = _FakeIg(
        [
            {
                "epic": "CS.D.EURUSD.MINI.IP",
                "instrumentName": "EUR/USD Mini",
                "marketStatus": "TRADEABLE",
            }
        ]
    )
    resolved = resolve_ticker_to_epic(ig, "EURUSD")
    assert resolved is not None
    assert resolved.source == "search"
    assert resolved.epic == "CS.D.EURUSD.MINI.IP"
