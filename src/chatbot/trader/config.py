from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Never surface these in dashboards / run lists.
_CONFIG_SECRET_KEYS = frozenset(
    {
        "ig_api_key",
        "ig_username",
        "ig_password",
        "ig_account_id",
        "fundmanager_token",
        "fundmanager_url",
    }
)

# Prefer this order in the unfold panel (unknown keys appended alphabetically).
_CONFIG_DISPLAY_ORDER = (
    "backtest_period",
    "llm_mode",
    "llm_temperature",
    "llm_trigger_mode",
    "llm_level_band_points",
    "llm_every_n",
    "llm_every_unit",
    "llm_every_bars",
    "max_open_positions",
    "order_size",
    "spread_points",
    "slippage_points",
    "hedge_beyond_entry_points",
    "allow_market_orders",
    "prevent_loss_exits",
    "flatten_before_close",
    "flatten_lead_minutes",
    "lookback_15m",
    "lookback_1h",
    "lookback_1d",
    "warmup_bars",
    "chart_show_rsi",
    "chart_show_pivots",
    "chart_pivot_period",
    "gemini_model",
    "data_timezone",
    "intrabar_pessimistic",
    "point_value",
    "pnl_currency",
    "overnight_funding_rate",
    "symbol",
    "strategy_name",
    "bot_id",
    "prompt_version",
)


def public_config_snapshot(data: dict[str, Any] | None) -> dict[str, Any]:
    """Strip secrets and order keys for UI display."""
    if not data:
        return {}
    cleaned = {k: v for k, v in data.items() if k not in _CONFIG_SECRET_KEYS}
    ordered: dict[str, Any] = {}
    for key in _CONFIG_DISPLAY_ORDER:
        if key in cleaned:
            ordered[key] = cleaned.pop(key)
    for key in sorted(cleaned):
        ordered[key] = cleaned[key]
    return ordered


@dataclass
class TraderConfig:
    """Shared live / backtest configuration."""

    symbol: str = "CAC40"
    epic: str = "IX.D.CAC.BMU.IP"
    max_open_positions: int = 4
    order_size: float = 1.0
    allow_market_orders: bool = False
    # When True, RiskGate + fills reject exits that would realize PnL ≤ 0 after spread.
    prevent_loss_exits: bool = False
    # Directional flatten before weekend / Euronext holiday IG close (hedge, don't scratch).
    flatten_before_close: bool = True
    flatten_lead_minutes: int = 30
    market_close_paris: str = "22:00"
    llm_every_bars: int = 24  # resolved 15m-bar stride (engine); default = 6h
    llm_every_n: int = 6  # UI: call every N units (interval mode only)
    llm_every_unit: str = "1h"  # UI: 15m | 1h
    llm_mode: str = "live"  # live | replay | charts_only
    # Gemini sampling temperature for live decisions (0 = more deterministic, 1 = more varied).
    llm_temperature: float = 0.0
    # levels = call on S/R approach·break / bootstrap / fill; interval = every N bars
    llm_trigger_mode: str = "levels"
    llm_level_band_points: float = 15.0
    spread_points: float = 1.5
    slippage_points: float = 0.0
    # Min distance (IG POINTS) for hedge_cover beyond entry/fill. Scaled by
    # price (FX 0.0001, indices 1.0). Default 2 (= IG min stop). Same-level
    # hedges are nudged, not rejected.
    hedge_beyond_entry_points: float = 2.0
    overnight_funding_rate: float = 0.0001  # per night on notional
    # Account currency per 1.0 price-unit move per lot (IG-derived when live).
    point_value: float = 1.0
    # ISO currency for PnL display (EUR/USD/…); empty until resolved.
    pnl_currency: str = ""
    timeframe: str = "15m"
    # Bars shown on each LLM chart (from full history as-of current bar).
    lookback_15m: int = 96
    lookback_1h: int = 72
    lookback_1d: int = 60
    # RSI period + min 15m history before first LLM call.
    warmup_bars: int = 14
    # Chart overlays (sent to LLM as PNG).
    chart_show_rsi: bool = True
    chart_show_pivots: bool = True
    # Traditional pivot session: D=Daily, W=Weekly, M=Monthly (not drawn on 1D charts).
    chart_pivot_period: str = "D"
    ig_acc_type: str = "DEMO"  # DEMO | LIVE
    ig_api_key: str = ""
    ig_username: str = ""
    ig_password: str = ""
    ig_account_id: str = ""
    fundmanager_url: str = ""
    fundmanager_token: str = ""
    bot_id: str = "evnor-cac-demo"
    strategy_name: str = "mean_reversion"
    gemini_model: str = "gemini-2.5-flash"
    prompt_version: str = "v1"
    # Tenant Config prompt (trading system prompt). Empty → profile default file.
    system_prompt: str = ""
    market_profile: str = "cac40"
    calendar_id: str = ""  # empty → derive from market_profile
    intrabar_pessimistic: bool = True
    data_timezone: str = "Europe/Paris"
    # Backtest window relative to last bar: 1w|2w|1m|3m|6m|1y|all
    backtest_period: str = "1w"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TraderConfig:
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        # Prefer n+unit when present; keep llm_every_bars in sync for the engine.
        if "llm_every_n" in data or "llm_every_unit" in data:
            cfg.llm_every_bars = cfg.resolve_llm_every_bars()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_snapshot(self) -> dict[str, Any]:
        """Config safe to show in UI (no credentials / tokens)."""
        return public_config_snapshot(self.to_dict())

    def resolve_llm_every_bars(self) -> int:
        """Convert UI rate (every N × 15m or N × 1h) into a 15m bar stride."""
        n = max(1, int(self.llm_every_n or 1))
        unit = (self.llm_every_unit or "15m").strip().lower()
        if unit in ("1h", "h", "hour", "hours"):
            return n * 4
        return n

    @classmethod
    def llm_rate_from_form(cls, *, every_n: int, unit: str) -> tuple[int, str, int]:
        """Return (llm_every_n, llm_every_unit, llm_every_bars)."""
        n = max(1, int(every_n or 1))
        u = (unit or "15m").strip().lower()
        if u in ("1h", "h", "hour", "hours"):
            u = "1h"
            bars = n * 4
        else:
            u = "15m"
            bars = n
        return n, u, bars


@dataclass
class LastLevels:
    support: float | None = None
    resistance: float | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"support": self.support, "resistance": self.resistance, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LastLevels:
        raw = dict(data or {})
        support = raw.get("support")
        resistance = raw.get("resistance")
        return cls(
            support=float(support) if support is not None else None,
            resistance=float(resistance) if resistance is not None else None,
            source=str(raw.get("source") or ""),
        )
