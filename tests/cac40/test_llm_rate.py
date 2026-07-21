from chatbot.cac40.config import Cac40Config


def test_llm_rate_from_form_15m() -> None:
    n, unit, bars = Cac40Config.llm_rate_from_form(every_n=3, unit="15m")
    assert (n, unit, bars) == (3, "15m", 3)


def test_llm_rate_from_form_hours() -> None:
    n, unit, bars = Cac40Config.llm_rate_from_form(every_n=2, unit="1h")
    assert (n, unit, bars) == (2, "1h", 8)


def test_resolve_llm_every_bars() -> None:
    cfg = Cac40Config(llm_every_n=1, llm_every_unit="1h")
    assert cfg.resolve_llm_every_bars() == 4
    cfg = Cac40Config(llm_every_n=4, llm_every_unit="15m")
    assert cfg.resolve_llm_every_bars() == 4


def test_defaults_one_week_and_six_hours() -> None:
    cfg = Cac40Config()
    assert cfg.backtest_period == "1w"
    assert cfg.llm_every_n == 6
    assert cfg.llm_every_unit == "1h"
    assert cfg.resolve_llm_every_bars() == 24
