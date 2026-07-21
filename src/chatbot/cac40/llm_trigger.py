from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Zone = Literal["mid_range", "in_band", "beyond"]

TRIGGER_BOOTSTRAP = "bootstrap"
TRIGGER_SUPPORT_APPROACH = "support_approach"
TRIGGER_SUPPORT_BREAK = "support_break"
TRIGGER_RESISTANCE_APPROACH = "resistance_approach"
TRIGGER_RESISTANCE_BREAK = "resistance_break"
TRIGGER_BOOK_CHANGE = "book_change"
TRIGGER_INTERVAL = "interval"

_FILL_EVENT_TYPES = frozenset({"open", "close"})


@dataclass(frozen=True)
class TriggerDecision:
    should_call: bool
    reasons: tuple[str, ...] = ()


def _f(bar: dict[str, Any], key: str) -> float:
    return float(bar[key])


def classify_support_zone(bar: dict[str, Any], support: float, band: float) -> Zone:
    """Geometric zone for the support side."""
    lo, hi, cl = _f(bar, "low"), _f(bar, "high"), _f(bar, "close")
    if lo < support - band:
        return "beyond"
    # Intersects [S-band, S+band]; close above band → left the zone (mid).
    if lo <= support + band and hi >= support - band and cl <= support + band:
        return "in_band"
    return "mid_range"


def classify_resistance_zone(bar: dict[str, Any], resistance: float, band: float) -> Zone:
    """Geometric zone for the resistance side."""
    lo, hi, cl = _f(bar, "low"), _f(bar, "high"), _f(bar, "close")
    if hi > resistance + band:
        return "beyond"
    if lo <= resistance + band and hi >= resistance - band and cl >= resistance - band:
        return "in_band"
    return "mid_range"


def _dedupe(reasons: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return tuple(out)


@dataclass
class LlmTrigger:
    """
    Gate LLM calls on bootstrap / S·R approach·break / book change.

    Zone state advances only after a successful call (on_success).
    Failures keep pending reasons so the next bar retries.
    """

    band_points: float = 15.0
    mode: str = "levels"  # levels | interval
    every_bars: int = 24

    support_zone: Zone = "mid_range"
    resistance_zone: Zone = "mid_range"
    pending_reasons: list[str] = field(default_factory=list)
    book_change_pending: bool = False
    _levels_key: tuple[float | None, float | None] | None = None
    _prev_position_ids: set[str] | None = None

    def note_fills(self, events: list[dict[str, Any]] | None) -> None:
        """Mark book_change when ledger reports open/close fills."""
        if not events:
            return
        for ev in events:
            if str(ev.get("type") or "") in _FILL_EVENT_TYPES:
                self.book_change_pending = True
                return

    def note_position_ids(self, position_ids: set[str] | list[str] | None) -> None:
        """Live helper: detect fills via position id set changes."""
        ids = set(position_ids or ())
        if self._prev_position_ids is not None and ids != self._prev_position_ids:
            self.book_change_pending = True
        self._prev_position_ids = ids

    def evaluate(
        self,
        *,
        bar: dict[str, Any],
        support: float | None,
        resistance: float | None,
        bar_index: int = 0,
    ) -> TriggerDecision:
        mode = (self.mode or "levels").strip().lower()
        if mode == "interval":
            every = max(1, int(self.every_bars or 1))
            should = bar_index % every == 0
            reasons = (TRIGGER_INTERVAL,) if should else ()
            if should:
                self.pending_reasons = list(reasons)
            return TriggerDecision(should_call=should, reasons=reasons)

        band = max(0.0, float(self.band_points))
        reasons: list[str] = list(self.pending_reasons)

        if support is None or resistance is None:
            reasons.append(TRIGGER_BOOTSTRAP)
            self.pending_reasons = list(_dedupe(reasons))
            return TriggerDecision(should_call=True, reasons=_dedupe(reasons))

        # New levels from outside (e.g. RiskGate) → recompute zones, no fire.
        key = (float(support), float(resistance))
        if self._levels_key != key:
            self._levels_key = key
            self.support_zone = classify_support_zone(bar, float(support), band)
            self.resistance_zone = classify_resistance_zone(bar, float(resistance), band)

        if self.book_change_pending:
            reasons.append(TRIGGER_BOOK_CHANGE)

        obs_s = classify_support_zone(bar, float(support), band)
        obs_r = classify_resistance_zone(bar, float(resistance), band)

        if self.support_zone == "mid_range" and obs_s == "in_band":
            reasons.append(TRIGGER_SUPPORT_APPROACH)
        if self.support_zone != "beyond" and obs_s == "beyond":
            reasons.append(TRIGGER_SUPPORT_BREAK)

        if self.resistance_zone == "mid_range" and obs_r == "in_band":
            reasons.append(TRIGGER_RESISTANCE_APPROACH)
        if self.resistance_zone != "beyond" and obs_r == "beyond":
            reasons.append(TRIGGER_RESISTANCE_BREAK)

        # Passive reset: close back mid-range re-arms that side without an LLM call.
        if obs_s == "mid_range" and self.support_zone != "mid_range":
            self.support_zone = "mid_range"
            reasons = [r for r in reasons if not r.startswith("support_")]
            self.pending_reasons = [r for r in self.pending_reasons if not r.startswith("support_")]
        if obs_r == "mid_range" and self.resistance_zone != "mid_range":
            self.resistance_zone = "mid_range"
            reasons = [r for r in reasons if not r.startswith("resistance_")]
            self.pending_reasons = [
                r for r in self.pending_reasons if not r.startswith("resistance_")
            ]

        deduped = _dedupe(reasons)
        if deduped:
            self.pending_reasons = list(deduped)
            return TriggerDecision(should_call=True, reasons=deduped)
        self.pending_reasons = []
        return TriggerDecision(should_call=False, reasons=())

    def on_success(
        self,
        *,
        bar: dict[str, Any],
        support: float | None,
        resistance: float | None,
    ) -> None:
        """Advance zone state after a successful LLM + gate cycle."""
        self.pending_reasons = []
        self.book_change_pending = False
        band = max(0.0, float(self.band_points))
        if support is None or resistance is None:
            self._levels_key = (support, resistance)
            self.support_zone = "mid_range"
            self.resistance_zone = "mid_range"
            return
        self._levels_key = (float(support), float(resistance))
        self.support_zone = classify_support_zone(bar, float(support), band)
        self.resistance_zone = classify_resistance_zone(bar, float(resistance), band)

    def on_failure(self) -> None:
        """Keep pending reasons / book_change for retry on the next bar."""
        # pending_reasons already set in evaluate; nothing else to clear.
        return
