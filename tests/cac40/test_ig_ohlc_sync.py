from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from chatbot.application.cac40_backtest_service import (
    MAX_OHLC_GAP_DAYS,
    default_ohlc_path,
    read_ohlc_sync_status,
    read_ohlc_worker_status,
    run_due_ig_ohlc_syncs,
    sync_ohlc_from_ig,
)
from chatbot.cac40.ig_connector import IgConnector, _ig_api_ts, _prices_payload_to_df
from chatbot.cac40.ig_ohlc import ig_config_from_connector
from chatbot.cac40.ohlc_store import load_ohlc_csv
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


@patch("chatbot.cac40.ig_ohlc.fetch_ig_ohlc_range")
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


@patch("chatbot.cac40.ig_ohlc.fetch_ig_ohlc_range")
def test_sync_up_to_date_adds_zero(mock_fetch, tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    existing = _sample_bars("2024-06-01 09:00:00", n=2)
    _write_csv(settings, "bot", existing)
    mock_fetch.return_value = existing.iloc[0:0].copy()
    now = datetime(2024, 6, 1, 10, 0, tzinfo=UTC)
    info = sync_ohlc_from_ig(
        settings,
        "bot",
        ig_config={"api_key": "k", "username": "u", "password": "p"},
        now=now,
    )
    assert info["added"] == 0
    assert info["bars"] == 2


@patch("chatbot.application.cac40_backtest_service.sync_ohlc_from_ig")
def test_run_due_skips_missing_csv_and_connector(mock_sync, tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    session = MagicMock()

    tenant = MagicMock()
    tenant.id = 1
    tenant.slug = "cac-bot"

    integration = MagicMock()
    integration.tenant_id = 1

    with (
        patch(
            "chatbot.adapters.persistence.integration_repository.SqlAlchemyIntegrationRepository"
        ) as mock_int_repo,
        patch(
            "chatbot.adapters.persistence.tenant_repository.SqlAlchemyTenantRepository"
        ),
        patch("chatbot.application.tenant_service.TenantService") as mock_tenant_svc,
        patch(
            "chatbot.adapters.persistence.connector_repository.SqlAlchemyConnectorRepository"
        ),
        patch("chatbot.application.connector_service.ConnectorService") as mock_conn_svc,
        patch(
            "chatbot.application.cac40_live_service.resolve_primary_ig_config"
        ) as mock_primary,
    ):
        mock_int_repo.return_value.list_active_by_type.return_value = [integration]
        mock_tenant_svc.return_value.get_by_id.return_value = tenant
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
    from chatbot.cac40.config import Cac40Config

    cfg = Cac40Config(ig_api_key="k", ig_username="u", ig_password="p")
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
    monkeypatch.setattr("chatbot.cac40.ig_connector.time.sleep", lambda *_: None)

    df = conn.fetch_ohlc_range(
        start=pd.Timestamp("2024-06-01T08:00:00Z"),
        end=pd.Timestamp("2024-06-01T09:00:00Z"),
        page_wait_seconds=0,
    )
    assert len(df) == 2
    conn.close()


def test_max_gap_constant() -> None:
    assert MAX_OHLC_GAP_DAYS == 60
