from pathlib import Path

from chatbot.application.cac40_backtest_service import ohlc_info, save_ohlc_upload
import pandas as pd

from chatbot.cac40.ohlc_store import load_ohlc_csv, slice_ohlc_period
from chatbot.config.settings import Settings


def test_save_ohlc_upload(tmp_path: Path):
    settings = Settings(data_root=tmp_path)
    csv = (
        b"Date,Open,High,Low,Close,Volume\n"
        b"2024-01-02 09:00:00+01:00,7500,7505,7495,7501,100\n"
        b"2024-01-02 09:15:00+01:00,7501,7508,7498,7504,120\n"
    )
    info = save_ohlc_upload(settings, "demo-bot", filename="cac.csv", content=csv)
    assert info["exists"] is True
    assert info["bars"] == 2
    assert Path(info["path"]).exists()
    again = ohlc_info(settings, "demo-bot")
    assert again["bars"] == 2


def test_save_ohlc_upload_backtestmarket(tmp_path: Path):
    settings = Settings(data_root=tmp_path)
    csv = (
        b"29/05/2000;02:00:00;6172;6200;6172;6200;417\n"
        b"29/05/2000;02:15:00;6199;6199;6179.5;6191.5;4234\n"
        b"29/05/2000;02:30:00;6196;6199;6194;6194;1080\n"
    )
    info = save_ohlc_upload(
        settings,
        "demo-bot",
        filename="mx-15m.csv",
        content=csv,
        source="backtestmarket",
    )
    assert info["exists"] is True
    assert info["bars"] == 3
    assert info["upload_source"] == "backtestmarket"
    df = load_ohlc_csv(Path(info["path"]), source="evenor")
    assert len(df) == 3
    assert float(df.iloc[0]["open"]) == 6172.0


def test_slice_ohlc_period_relative_to_end() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="Europe/Paris")
    df = pd.DataFrame(
        {
            "open": range(120),
            "high": range(120),
            "low": range(120),
            "close": range(120),
        },
        index=idx,
    )
    assert len(slice_ohlc_period(df, "all")) == 120
    week = slice_ohlc_period(df, "1w")
    assert len(week) < 120
    assert week.index.max() == df.index.max()
    assert week.index.min() >= df.index.max() - pd.Timedelta(weeks=1)
    month = slice_ohlc_period(df, "1m")
    assert len(month) < 120
    assert month.index.max() == df.index.max()
    assert month.index.min() >= df.index.max() - pd.DateOffset(months=1)
