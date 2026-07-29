"""IG dealingRules POINTS → price units (shared by RiskGate + IgConnector)."""

from __future__ import annotations


def infer_point_size(price: float) -> float:
    """Price value of one IG POINTS unit when scalingFactor is unknown.

    FX majors (~1.x): 0.0001. Mid-priced: 0.01. Indices: 1.0.
    """
    mid = float(price or 0.0)
    if 0 < mid < 10:
        return 0.0001
    if 0 < mid < 500:
        return 0.01
    return 1.0
