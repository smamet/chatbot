from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

FX_CACHE_FILENAME = ".fx-rates-cache.json"
FX_CACHE_TTL_HOURS = 24
USD_PIVOT = "USD"
FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest?from=USD"
OPEN_ER_API_URL = "https://open.er-api.com/v6/latest/USD"


def fx_cache_path(data_root: Path) -> Path:
    return data_root / FX_CACHE_FILENAME


def _normalize_currency(code: str) -> str:
    return str(code or "").strip().upper()


def _parse_rates_payload(payload: dict[str, Any]) -> dict[str, float] | None:
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        return None
    clean: dict[str, float] = {USD_PIVOT: 1.0}
    for code, value in rates.items():
        ccy = _normalize_currency(str(code))
        if not ccy:
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if rate > 0:
            clean[ccy] = rate
    return clean if len(clean) > 1 else None


def _fetch_frankfurter_rates() -> dict[str, float] | None:
    try:
        with urlopen(FRANKFURTER_URL, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Frankfurter FX fetch failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_rates_payload(payload)


def _fetch_open_er_api_rates() -> dict[str, float] | None:
    try:
        with urlopen(OPEN_ER_API_URL, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("open.er-api FX fetch failed: %s", exc)
        return None
    if not isinstance(payload, dict) or payload.get("result") != "success":
        return None
    return _parse_rates_payload(payload)


def _fetch_live_rates() -> dict[str, float] | None:
    rates = _fetch_frankfurter_rates()
    if rates:
        return rates
    return _fetch_open_er_api_rates()


def read_fx_cache(data_root: Path) -> dict[str, float] | None:
    path = fx_cache_path(data_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    fetched_at = str(payload.get("fetched_at") or "").strip()
    rates = payload.get("rates")
    if not fetched_at or not isinstance(rates, dict):
        return None
    try:
        cached_time = datetime.fromisoformat(fetched_at)
    except ValueError:
        return None
    if cached_time.tzinfo is None:
        cached_time = cached_time.replace(tzinfo=UTC)
    if datetime.now(UTC) - cached_time > timedelta(hours=FX_CACHE_TTL_HOURS):
        return None
    clean: dict[str, float] = {}
    for code, value in rates.items():
        ccy = _normalize_currency(str(code))
        if not ccy:
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if rate > 0:
            clean[ccy] = rate
    return clean if clean else None


def write_fx_cache(data_root: Path, rates: dict[str, float]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "base": USD_PIVOT,
        "rates": {USD_PIVOT: 1.0, **{k: v for k, v in rates.items() if k != USD_PIVOT}},
    }
    fx_cache_path(data_root).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


class FxRateService:
    """FX rates with USD pivot, 24h file cache under DATA_ROOT."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._rates: dict[str, float] | None = None

    def get_rates_usd_pivot(self) -> dict[str, float]:
        if self._rates is not None:
            return self._rates
        cached = read_fx_cache(self._data_root)
        if cached:
            self._rates = cached
            return cached
        live = _fetch_live_rates()
        if live:
            write_fx_cache(self._data_root, live)
            self._rates = live
            return live
        logger.warning("FX rates unavailable; conversions will be skipped")
        self._rates = {}
        return {}

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> float | None:
        if amount <= 0:
            return None
        src = _normalize_currency(from_ccy)
        dst = _normalize_currency(to_ccy)
        if not src or not dst:
            return None
        if src == dst:
            return amount
        rates = self.get_rates_usd_pivot()
        if not rates:
            return None
        from_rate = rates.get(src)
        to_rate = rates.get(dst)
        if from_rate is None or to_rate is None or from_rate <= 0:
            return None
        # rates[ccy] = units of ccy per 1 USD
        usd_amount = amount / from_rate
        return usd_amount * to_rate
