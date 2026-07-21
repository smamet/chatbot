from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from chatbot.cac40.models import LlmAction, LlmAnalysis, LlmDecision, MarketSnapshot

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """You are a discretionary CAC40 mean-reversion trader analyzing candlestick charts.

Book continuity:
- Read snapshot.positions and snapshot.working_orders first.
- If a plan is already working, prefer hold with empty actions or amend/cancel only.
- Do not place hedge_cover/tp before a primary exists; attach position_id after fills.
- Size must equal order_size. Prefer fewer actions.

Rules:
- Identify support and resistance from the charts (15m execution, 1H and Daily context).
- Prefer LIMIT entries at support (BUY) / resistance (SELL) and LIMIT take-profits.
- Place STOP hedge covers only to protect existing legs.
- Do not use market orders unless explicitly told.
- Output STRICT JSON only, no markdown.

JSON schema:
{
  "analysis": {
    "support": number,
    "resistance": number,
    "bias": "long_from_support|short_from_resistance|hold|breakout",
    "rsi_note": "string",
    "pivot_note": "string"
  },
  "actions": [
    {"op": "place_limit|place_stop|amend_order|cancel_order|market_open|market_close",
     "side": "BUY|SELL", "level": number, "size": number, "purpose": "entry|tp|hedge_cover|close",
     "order_id": "optional", "position_id": "optional", "reason": "string"}
  ]
}
"""


def load_prompt(prompt_path: Path | None = None) -> str:
    if prompt_path and prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    default = Path(__file__).resolve().parents[3] / "prompts" / "cac40" / "system.md"
    if default.exists():
        return default.read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def summarize_decision(decision: LlmDecision | None) -> dict[str, Any] | None:
    """Compact prior-cycle summary for continuity in the next LLM call."""
    if decision is None:
        return None
    analysis = decision.analysis
    actions_summary: list[dict[str, Any]] = []
    for a in decision.actions[:12]:
        actions_summary.append(
            {
                "op": a.op,
                "side": a.side,
                "level": a.level,
                "purpose": a.purpose,
                "order_id": a.order_id,
                "position_id": a.position_id,
            }
        )
    return {
        "bias": analysis.bias,
        "support": analysis.support,
        "resistance": analysis.resistance,
        "actions": actions_summary,
    }


def build_user_payload(
    snapshot: MarketSnapshot,
    phase: str,
    *,
    order_size: float = 1.0,
    max_open_positions: int = 4,
    last_decision: dict[str, Any] | None = None,
    allow_market_orders: bool = False,
) -> str:
    instructions = [
        "Manage the existing book first. Do not duplicate entries already in working_orders.",
        "Prefer empty actions or amend/cancel when the prior plan is still valid.",
        f"Use size={order_size} on every place_* action (never larger aggregates).",
        f"Respect max_open_positions={max_open_positions}.",
    ]
    if allow_market_orders:
        instructions.append("Market orders are allowed when needed to flatten.")
    else:
        instructions.append("Propose working LIMIT/STOP orders only. Do not use market_* ops.")

    payload: dict[str, Any] = {
        "phase": phase,
        "order_size": order_size,
        "max_open_positions": max_open_positions,
        "snapshot": snapshot.to_dict(),
        "instructions": " ".join(instructions),
    }
    if last_decision:
        payload["last_decision"] = last_decision
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_llm_json(text: str) -> LlmDecision:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    analysis_raw = data.get("analysis") or {}
    analysis = LlmAnalysis(
        support=_maybe_float(analysis_raw.get("support")),
        resistance=_maybe_float(analysis_raw.get("resistance")),
        bias=str(analysis_raw.get("bias") or "hold"),
        rsi_note=str(analysis_raw.get("rsi_note") or ""),
        pivot_note=str(analysis_raw.get("pivot_note") or ""),
    )
    actions = [LlmAction.from_dict(a) for a in data.get("actions") or []]
    return LlmDecision(analysis=analysis, actions=actions, raw=data)


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


class GeminiDecisionClient:
    """Thin multimodal Gemini wrapper. Fail-closed on parse errors."""

    def __init__(self, *, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.last_error: str | None = None

    def decide(
        self,
        *,
        images: dict[str, bytes],
        snapshot: MarketSnapshot,
        phase: str,
        prompt: str | None = None,
        order_size: float = 1.0,
        max_open_positions: int = 4,
        last_decision: dict[str, Any] | None = None,
        allow_market_orders: bool = False,
    ) -> LlmDecision | None:
        self.last_error = None
        if not self.api_key:
            self.last_error = "Gemini API key missing"
            logger.error(self.last_error)
            return None
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            self.last_error = "google-genai not installed"
            logger.error(self.last_error)
            return None

        client = genai.Client(api_key=self.api_key)
        parts: list[Any] = [types.Part.from_text(text=prompt or DEFAULT_PROMPT)]
        parts.append(
            types.Part.from_text(
                text=build_user_payload(
                    snapshot,
                    phase,
                    order_size=order_size,
                    max_open_positions=max_open_positions,
                    last_decision=last_decision,
                    allow_market_orders=allow_market_orders,
                )
            )
        )
        for tf, png in images.items():
            parts.append(types.Part.from_text(text=f"Chart timeframe: {tf}"))
            parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=parts,
            )
            text = getattr(response, "text", None) or ""
            if not text and getattr(response, "candidates", None):
                text = response.candidates[0].content.parts[0].text  # type: ignore[index]
            return parse_llm_json(text)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("LLM decision failed (%s)", self.model)
            return None
