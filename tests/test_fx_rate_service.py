from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from evenor.application.fx_rate_service import (
    FxRateService,
    read_fx_cache,
    write_fx_cache,
)


def test_write_and_read_fx_cache_within_ttl(tmp_path: Path) -> None:
    rates = {"USD": 1.0, "MUR": 46.07, "EUR": 0.92}
    write_fx_cache(tmp_path, rates)
    loaded = read_fx_cache(tmp_path)
    assert loaded is not None
    assert loaded["MUR"] == 46.07


def test_read_fx_cache_expired(tmp_path: Path) -> None:
    path = tmp_path / ".fx-rates-cache.json"
    old = datetime.now(UTC) - timedelta(hours=25)
    payload = {
        "fetched_at": old.isoformat(),
        "base": "USD",
        "rates": {"USD": 1.0, "MUR": 40.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_fx_cache(tmp_path) is None


def test_fx_rate_service_convert_via_usd_pivot(tmp_path: Path) -> None:
    write_fx_cache(tmp_path, {"USD": 1.0, "MUR": 50.0, "EUR": 0.5})
    fx = FxRateService(tmp_path)
    # 100 MUR -> 2 USD -> 1 EUR (rates: MUR=50/USD, EUR=0.5/USD)
    assert fx.convert(100.0, "MUR", "EUR") == 1.0
    assert fx.convert(10.0, "USD", "MUR") == 500.0
    assert fx.convert(10.0, "MUR", "MUR") == 10.0


def test_fx_rate_service_convert_unknown_currency(tmp_path: Path) -> None:
    write_fx_cache(tmp_path, {"USD": 1.0, "MUR": 50.0})
    fx = FxRateService(tmp_path)
    assert fx.convert(10.0, "MUR", "XYZ") is None


def test_fx_rate_service_fetches_live_when_cache_missing(tmp_path: Path) -> None:
    fx = FxRateService(tmp_path)
    with patch(
        "evenor.application.fx_rate_service._fetch_live_rates",
        return_value={"USD": 1.0, "MUR": 45.0},
    ):
        rates = fx.get_rates_usd_pivot()
    assert rates["MUR"] == 45.0
    assert read_fx_cache(tmp_path) is not None
