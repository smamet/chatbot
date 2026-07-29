from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from chatbot.application.trader_backtest_service import (
    MAX_OHLC_GAP_DAYS,
    default_ohlc_path,
    read_ohlc_sync_status,
    read_ohlc_worker_status,
    run_due_ig_ohlc_syncs,
    sync_ohlc_from_ig,
)
from chatbot.trader.ig_connector import IgConnector, _ig_api_ts, _prices_payload_to_df
from chatbot.trader.ig_ohlc import ig_config_from_connector
from chatbot.trader.ohlc_store import load_ohlc_csv
from chatbot.config.settings import Settings


def _sample_bars(start: str, n: int = 3) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min", tz="Europe/Paris")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [10] * n,
        },
        index=idx,
    )


def _write_csv(settings: Settings, slug: str, df: pd.DataFrame) -> Path:
    path = default_ohlc_path(settings, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out.to_csv(path, index=False)
    return path


def test_ig_config_from_connector_maps_keys() -> None:
    cfg = ig_config_from_connector(
        {
            "api_key": "k",
            "username": "u",
            "password": "p",
            "account_id": "ACC",
            "acc_type": "demo",
            "epic": "IX.D.CAC.DAILY.IP",
        }
    )
    assert cfg.ig_api_key == "k"
    assert cfg.ig_username == "u"
    assert cfg.ig_password == "p"
    assert cfg.ig_account_id == "ACC"
    assert cfg.ig_acc_type == "DEMO"
    assert cfg.epic == "IX.D.CAC.DAILY.IP"


def test_prices_payload_to_df_mids() -> None:
    prices = [
        {
            "snapshotTimeUTC": "2024-06-01T08:00:00",
            "openPrice": {"bid": 10.0, "ask": 12.0},
            "highPrice": {"bid": 13.0, "ask": 15.0},
            "lowPrice": {"bid": 9.0, "ask": 11.0},
            "closePrice": {"bid": 11.0, "ask": 13.0},
            "lastTradedVolume": 5,
        }
    ]
    df = _prices_payload_to_df(prices)
    assert len(df) == 1
    assert float(df.iloc[0]["open"]) == 11.0
    assert float(df.iloc[0]["close"]) == 12.0


def test_ig_api_ts_format() -> None:
    ts = pd.Timestamp("2024-06-01 10:00:00", tz="Europe/Paris")
    assert _ig_api_ts(ts) == "2024-06-01T08:00:00"


def test_sync_gap_too_large_raises(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    old = _sample_bars("2024-01-02 09:00:00")
    _write_csv(settings, "bot", old)
    now = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="gap"):
        sync_ohlc_from_ig(
            settings,
            "bot",
            ig_config={"api_key": "k", "username": "u", "password": "p"},
            now=now,
        )
    status = read_ohlc_sync_status(settings, "bot")
    assert status.get("ok") is False


def test_sync_bootstrap_disallowed_without_csv(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    with pytest.raises(ValueError, match="missing"):
        sync_ohlc_from_ig(
            settings,
            "bot",
            ig_config={"api_key": "k", "username": "u", "password": "p"},
            allow_bootstrap=False,
        )


@patch("chatbot.trader.ig_ohlc.fetch_ig_ohlc_range")
def test_sync_appends_only_new_bars(mock_fetch, tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    existing = _sample_bars("2024-06-01 09:00:00", n=2)
    _write_csv(settings, "bot", existing)
    fresh = _sample_bars("2024-06-01 09:30:00", n=2)
    # include a duplicate of last bar to ensure filter works
    overlap = existing.iloc[[-1]].copy()
    mock_fetch.return_value = pd.concat([overlap, fresh]).sort_index()

    now = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    info = sync_ohlc_from_ig(
        settings,
        "bot",
        ig_config={"api_key": "k", "username": "u", "password": "p"},
        trigger="manual",
        now=now,
    )
    assert info["added"] == 2
    assert info["bars"] == 4
    assert info["trigger"] == "manual"
    assert info["last_candle"]
    loaded = load_ohlc_csv(Path(info["path"]))
    assert len(loaded) == 4
    assert loaded.index[-1] == fresh.index[-1]
    status = read_ohlc_sync_status(settings, "bot")
    assert status["ok"] is True
    assert status["added"] == 2
    assert status["trigger"] == "manual"
    assert status["last_candle"]


@patch("chatbot.trader.ig_ohlc.fetch_ig_ohlc_range")
def test_sync_up_to_date_adds_zero(mock_fetch, tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    existing = _sample_bars("2024-06-01 09:00:00", n=2)  # last 09:15 Paris
    _write_csv(settings, "bot", existing)
    mock_fetch.return_value = existing.iloc[0:0].copy()
    # 09:20 Paris = 07:20 UTC → expected closed = 09:00; CSV last 09:15 is ahead/equal enough
    now = datetime(2024, 6, 1, 7, 20, tzinfo=UTC)
    info = sync_ohlc_from_ig(
        settings,
        "bot",
        ig_config={"api_key": "k", "username": "u", "password": "p"},
        now=now,
    )
    assert info["added"] == 0
    assert info["bars"] == 2
    assert info["ok"] is True


@patch("chatbot.application.trader_backtest_service.sync_ohlc_from_ig")
def test_run_due_skips_missing_csv_and_connector(mock_sync, tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    session = MagicMock()

    tenant = MagicMock()
    tenant.id = 1
    tenant.slug = "cac-bot"

    with (
        patch(
            "chatbot.adapters.persistence.tenant_repository.SqlAlchemyTenantRepository"
        ) as mock_tenant_repo,
        patch("chatbot.application.tenant_service.TenantService"),
        patch(
            "chatbot.adapters.persistence.connector_repository.SqlAlchemyConnectorRepository"
        ),
        patch("chatbot.application.connector_service.ConnectorService"),
        patch(
            "chatbot.application.trader_live_service.resolve_primary_ig_config"
        ) as mock_primary,
        patch("chatbot.application.trader_live_service.load_live_config") as mock_live,
    ):
        mock_tenant_repo.return_value.list_active_traders.return_value = [tenant]
        mock_live.return_value = {"mode": "off"}
        mock_primary.return_value = None

        logs = run_due_ig_ohlc_syncs(session, settings)
        assert any("no active IG connector" in line for line in logs)
        mock_sync.assert_not_called()

        mock_primary.return_value = {
            "api_key": "k",
            "username": "u",
            "password": "p",
        }
        logs = run_due_ig_ohlc_syncs(session, settings)
        assert any("OHLC CSV missing" in line for line in logs)
        mock_sync.assert_not_called()

        _write_csv(settings, "cac-bot", _sample_bars("2024-06-01 09:00:00"))
        mock_sync.return_value = {
            "added": 1,
            "bars": 3,
            "last_candle": "2024-06-01 09:30:00+02:00",
        }
        logs = run_due_ig_ohlc_syncs(session, settings)
        mock_sync.assert_called_once()
        assert mock_sync.call_args.kwargs["allow_bootstrap"] is False
        assert mock_sync.call_args.kwargs["trigger"] == "worker"
        assert any("added 1" in line for line in logs)
        worker = read_ohlc_worker_status(settings)
        assert worker.get("finished_at")
        assert worker.get("tenants_ok") == 1


def test_fetch_ohlc_range_pages(monkeypatch) -> None:
    from chatbot.trader.config import TraderConfig

    cfg = TraderConfig(ig_api_key="k", ig_username="u", ig_password="p")
    conn = IgConnector(cfg, dry_run=True)
    conn._cst = "cst"  # noqa: SLF001
    conn._security = "sec"  # noqa: SLF001

    page1 = {
        "prices": [
            {
                "snapshotTimeUTC": "2024-06-01T08:00:00",
                "openPrice": {"bid": 1, "ask": 1},
                "highPrice": {"bid": 1, "ask": 1},
                "lowPrice": {"bid": 1, "ask": 1},
                "closePrice": {"bid": 1, "ask": 1},
            }
        ],
        "metadata": {"pageData": {"pageNumber": 1, "totalPages": 2}},
    }
    page2 = {
        "prices": [
            {
                "snapshotTimeUTC": "2024-06-01T08:15:00",
                "openPrice": {"bid": 2, "ask": 2},
                "highPrice": {"bid": 2, "ask": 2},
                "lowPrice": {"bid": 2, "ask": 2},
                "closePrice": {"bid": 2, "ask": 2},
            }
        ],
        "metadata": {"pageData": {"pageNumber": 2, "totalPages": 2}},
    }
    responses = [page1, page2]

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.is_error = False
            self.request = MagicMock(url="https://demo-api.ig.com/gateway/deal/prices/x")

        def json(self):
            return self._payload

    def fake_get(url, headers=None, params=None):
        return FakeResp(responses.pop(0))

    monkeypatch.setattr(conn._client, "get", fake_get)  # noqa: SLF001
    monkeypatch.setattr("chatbot.trader.ig_connector.time.sleep", lambda *_: None)

    df = conn.fetch_ohlc_range(
        start=pd.Timestamp("2024-06-01T08:00:00Z"),
        end=pd.Timestamp("2024-06-01T09:00:00Z"),
        page_wait_seconds=0,
    )
    assert len(df) == 2
    conn.close()


@patch("chatbot.trader.ig_ohlc.fetch_ig_ohlc_range")
def test_sync_zero_bars_while_behind_raises(mock_fetch, tmp_path: Path) -> None:
    """Success-+0 while CSV is mid-session behind is a failure, not 'ok'."""
    settings = Settings(data_root=tmp_path)
    # Last bar early morning; "now" is afternoon same day.
    existing = _sample_bars("2026-07-22 06:00:00", n=2)  # 06:00, 06:15
    _write_csv(settings, "bot", existing)
    mock_fetch.return_value = existing.iloc[0:0].copy()
    now = datetime(2026, 7, 22, 12, 20, tzinfo=UTC)  # 14:20 Paris
    with pytest.raises(ValueError, match="0 new 15m bars"):
        sync_ohlc_from_ig(
            settings,
            "bot",
            ig_config={"api_key": "k", "username": "u", "password": "p"},
            now=now,
        )
    status = read_ohlc_sync_status(settings, "bot")
    assert status.get("ok") is False
    assert status.get("added") == 0


@patch("chatbot.trader.ig_ohlc.fetch_ig_ohlc_range")
def test_sync_appends_across_preopen_to_cash_open(mock_fetch, tmp_path: Path) -> None:
    """Overnight/pre-open last bar → cash-open bars must append (natural break)."""
    settings = Settings(data_root=tmp_path)
    existing = _sample_bars("2026-07-22 06:00:00", n=2)  # ends 06:15
    _write_csv(settings, "bot", existing)
    fresh = _sample_bars("2026-07-22 09:00:00", n=3)
    mock_fetch.return_value = fresh
    now = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
    info = sync_ohlc_from_ig(
        settings,
        "bot",
        ig_config={"api_key": "k", "username": "u", "password": "p"},
        now=now,
    )
    assert info["added"] == 3
    assert info["ok"] is True


def test_sync_via_catchup_max_fallback_after_overlap(tmp_path: Path) -> None:
    """Stuck pre-open CSV: range only overlaps last bar → max= tip must append."""
    settings = Settings(data_root=tmp_path)
    existing = _sample_bars("2026-07-22 06:00:00", n=2)  # ends 06:15
    _write_csv(settings, "bot", existing)
    overlap = existing.iloc[[-1]].copy()
    tip = _sample_bars("2026-07-22 09:00:00", n=4)

    class _FakeConn:
        last_price_allowance = {"remaining": 9000, "total": 10000}
        authenticated = True

        def login(self) -> None:
            return None

        def close(self) -> None:
            return None

        def fetch_ohlc_range(self, **kwargs):  # noqa: ANN003
            return overlap

        def get_ohlc(self, timeframe: str, n: int):
            return tip

    with (
        patch("chatbot.trader.ig_ohlc.IgConnector", return_value=_FakeConn()),
        patch("chatbot.trader.ig_ohlc.ig_config_from_connector") as mock_cfg,
    ):
        mock_cfg.return_value = MagicMock(
            ig_api_key="k", ig_username="u", ig_password="p"
        )
        now = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
        info = sync_ohlc_from_ig(
            settings,
            "bot",
            ig_config={"api_key": "k", "username": "u", "password": "p"},
            now=now,
        )
    assert info["added"] == 4
    assert info["ok"] is True
    loaded = load_ohlc_csv(Path(info["path"]))
    assert str(loaded.index[-1]).startswith("2026-07-22 09:45")


def test_natural_break_preopen_to_cash() -> None:
    from chatbot.trader.ohlc_store import is_natural_session_break

    a = pd.Timestamp("2026-07-22 06:15:00", tz="Europe/Paris")
    b = pd.Timestamp("2026-07-22 09:00:00", tz="Europe/Paris")
    assert is_natural_session_break(a, b) is True
    # True mid-session hole still flagged.
    c = pd.Timestamp("2026-07-22 10:00:00", tz="Europe/Paris")
    d = pd.Timestamp("2026-07-22 11:00:00", tz="Europe/Paris")
    assert is_natural_session_break(c, d) is False


def test_catchup_falls_back_to_max_when_range_empty() -> None:
    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgConnector
    from chatbot.trader.ig_ohlc import catchup_ohlc_15m

    cfg = TraderConfig()
    conn = IgConnector(cfg, dry_run=True)
    conn._cst = "cst"  # noqa: SLF001

    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    empty.index = pd.DatetimeIndex([], tz="Europe/Paris")
    tip = _sample_bars("2026-07-22 09:00:00", n=4)

    conn.fetch_ohlc_range = lambda **kwargs: empty  # type: ignore[method-assign]
    conn.get_ohlc = lambda tf, n: tip  # type: ignore[method-assign]

    df, mode = catchup_ohlc_15m(
        conn,
        start=pd.Timestamp("2026-07-22 06:15:00", tz="Europe/Paris"),
        end=pd.Timestamp("2026-07-22 12:00:00", tz="UTC"),
    )
    assert mode == "max_fallback"
    assert len(df) == 4
    conn.close()


def test_catchup_falls_back_when_range_only_has_overlap() -> None:
    """IG often re-sends the last candle; that must not skip max= fallback."""
    from chatbot.trader.config import TraderConfig
    from chatbot.trader.ig_connector import IgConnector
    from chatbot.trader.ig_ohlc import catchup_ohlc_15m

    cfg = TraderConfig()
    conn = IgConnector(cfg, dry_run=True)
    conn._cst = "cst"  # noqa: SLF001

    overlap = _sample_bars("2026-07-22 06:15:00", n=1)
    tip = _sample_bars("2026-07-22 09:00:00", n=4)
    conn.fetch_ohlc_range = lambda **kwargs: overlap  # type: ignore[method-assign]
    conn.get_ohlc = lambda tf, n: tip  # type: ignore[method-assign]

    df, mode = catchup_ohlc_15m(
        conn,
        start=pd.Timestamp("2026-07-22 06:15:00", tz="Europe/Paris"),
        end=pd.Timestamp("2026-07-22 12:00:00", tz="UTC"),
    )
    assert mode == "max_fallback"
    assert len(df) == 4
    assert df.index[0] > overlap.index[0]
    conn.close()


def test_natural_break_rejects_preopen_to_midday_splice() -> None:
    from chatbot.trader.ohlc_store import is_natural_session_break

    a = pd.Timestamp("2026-07-22 06:15:00", tz="Europe/Paris")
    midday = pd.Timestamp("2026-07-22 12:00:00", tz="Europe/Paris")
    assert is_natural_session_break(a, midday) is False


def test_max_gap_constant() -> None:
    assert MAX_OHLC_GAP_DAYS == 60
