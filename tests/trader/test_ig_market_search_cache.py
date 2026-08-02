from __future__ import annotations

from pathlib import Path

from chatbot.trader.epic_resolve import autocomplete_symbol_rows, rank_search_rows
from chatbot.trader.ig_market_search_cache import cached_search_markets


class _CountingIg:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls = 0
        self.config = type("C", (), {"ig_acc_type": "DEMO"})()

    def search_markets(self, term: str) -> list[dict]:
        self.calls += 1
        return list(self._rows)

    def resolve_account_type(self) -> str:
        return "CFD"


def test_cached_search_markets_hits_disk(tmp_path: Path) -> None:
    ig = _CountingIg(
        [
            {
                "epic": "CS.D.EURUSD.MINI.IP",
                "instrumentName": "EUR/USD Mini",
                "marketStatus": "EDITS_ONLY",
            }
        ]
    )
    rows1, from_cache1 = cached_search_markets(
        ig, "EUR", cache_dir=tmp_path, acc_type="DEMO"
    )
    rows2, from_cache2 = cached_search_markets(
        ig, "EUR", cache_dir=tmp_path, acc_type="DEMO"
    )
    assert not from_cache1 and from_cache2
    assert ig.calls == 1
    assert rows1[0]["epic"] == rows2[0]["epic"]


def test_autocomplete_uses_cache_and_accepts_non_tradeable(tmp_path: Path) -> None:
    ig = _CountingIg(
        [
            {
                "epic": "CS.D.EURUSD.MINI.IP",
                "instrumentName": "EUR/USD Mini",
                "marketStatus": "EDITS_ONLY",
            }
        ]
    )
    first = autocomplete_symbol_rows(ig, "EUR", cache_dir=tmp_path)
    second = autocomplete_symbol_rows(ig, "EUR", cache_dir=tmp_path)
    assert first["results"]
    assert first["results"][0]["epic"] == "CS.D.EURUSD.MINI.IP"
    assert first["ig_calls"] == 1
    assert second["cache_hits"] >= 1
    assert ig.calls == 1


def test_rank_accepts_edits_only_when_not_requiring_tradeable() -> None:
    rows = [
        {
            "epic": "CS.D.EURUSD.MINI.IP",
            "instrumentName": "EUR/USD Mini",
            "marketStatus": "EDITS_ONLY",
        }
    ]
    assert not rank_search_rows(rows, symbol="EUR", require_tradeable=True)
    ranked = rank_search_rows(rows, symbol="EUR", require_tradeable=False)
    assert ranked and ranked[0][1] == "CS.D.EURUSD.MINI.IP"
