"""Market profiles: defaults for symbol/epic/calendar/prompt per product."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketProfile:
    id: str
    label: str
    default_symbol: str
    default_epic: str
    calendar_id: str
    prompt_relative: str  # under prompts/
    # IG dealingRules POINTS beyond entry for hedge_cover auto-nudge.
    # Same default for FX and indices: 2 points (= IG min stop on EURUSD Mini).
    hedge_beyond_entry_points: float = 2.0
    # Account currency per 1.0 price-unit move per lot.
    # Indices ≈ €1/pt → 1.0. EURUSD Mini ≈ $1/pip (0.0001) → 10000.
    default_point_value: float = 1.0


PROFILES: dict[str, MarketProfile] = {
    "cac40": MarketProfile(
        id="cac40",
        label="CAC40 (France 40)",
        default_symbol="CAC40",
        default_epic="IX.D.CAC.BMU.IP",
        calendar_id="euronext_fr40",
        prompt_relative="trader/profiles/cac40.md",
        hedge_beyond_entry_points=2.0,
        default_point_value=1.0,
    ),
    "eurusd": MarketProfile(
        id="eurusd",
        label="EURUSD (FX)",
        default_symbol="EURUSD",
        default_epic="CS.D.EURUSD.MINI.IP",
        calendar_id="forex_ig",
        prompt_relative="trader/profiles/eurusd.md",
        hedge_beyond_entry_points=2.0,
        default_point_value=10000.0,
    ),
}


def get_profile(profile_id: str | None) -> MarketProfile:
    key = str(profile_id or "cac40").strip().lower() or "cac40"
    # Legacy alias from early rename pass
    if key == "trader":
        key = "cac40"
    return PROFILES.get(key) or PROFILES["cac40"]


def point_value_for_symbol(symbol: str | None, profile_id: str | None = None) -> float:
    """Resolve account-currency point value from profile or symbol hint."""
    if profile_id:
        return float(get_profile(profile_id).default_point_value)
    key = str(symbol or "").strip().upper()
    for profile in PROFILES.values():
        if profile.default_symbol.upper() == key:
            return float(profile.default_point_value)
    if "EURUSD" in key or "EUR/USD" in key:
        return float(PROFILES["eurusd"].default_point_value)
    return 1.0


def prompts_root() -> Path:
    return Path(__file__).resolve().parents[3] / "prompts"


def default_prompt_text(profile_id: str | None = None) -> str:
    profile = get_profile(profile_id)
    path = prompts_root() / profile.prompt_relative
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def list_profiles_for_ui() -> list[dict[str, str]]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "default_symbol": p.default_symbol,
            "default_epic": p.default_epic,
        }
        for p in PROFILES.values()
    ]
