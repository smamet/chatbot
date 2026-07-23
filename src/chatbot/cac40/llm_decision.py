from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.gemini_usage import usage_from_response
from chatbot.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository
from chatbot.application.usage_recorder_service import UsageRecorderService
from chatbot.cac40.models import LlmAction, LlmAnalysis, LlmDecision, MarketSnapshot

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]

DEFAULT_PROMPT = """You are a discretionary CAC40 mean-reversion trader analyzing candlestick charts.

Vocabulary: entry = resting working order. primary = open filled leg in snapshot.positions.

Profit-only exits (target ~100% win rate on closed trades):
- NEVER close a leg at a loss. Every TP/close must realize PnL > 0 after spread.
- Losing primary stays open under hedge protection; do not scratch both legs.
- Close a hedge only when it can exit in profit on mean reversion; then TP the primary in profit.
- Hedge→new S/R play: if a hedge is open in profit and charts show a fresh support buy / resistance sell, do NOT pyramid another hedge_cover for the old primary. Same decision: close/TP the profitable hedge, cancel pending further hedge stops it replaces, place new LIMIT entry at that S/R (prefer the filled hedge level when that is the level you play) + tp + reverse-side hedge_cover a few points beyond. Keep the underwater primary open.
- Pyramid further hedge_cover only while price is still breaking with no clear S/R bounce.
- Avoid flat/scratch exits that only pay the spread.

Book continuity:
- Read snapshot.positions and snapshot.working_orders first.
- If a plan is already working, prefer hold with empty actions or amend/cancel only.
- New entry MUST include tp + reverse-side hedge_cover STOP in the SAME decision. TP attaches on the entry (arms on fill). hedge_cover is NOT a stop-loss — it force-opens an opposing hedge leg and is placed on IG immediately with the entry.
- After a primary exists with no new S/R entry idea: manage with position_id; pyramid hedge_cover only while breaking.
- Size must equal order_size. Prefer fewer actions; set the full bracket at once.

Weekend / holiday gap protection (CRITICAL):
- When market_clock.flatten_now is true, the book MUST be directionally flat (net size 0) before close.
- market_open the opposite side with size = |market_clock.net_exposure| and purpose hedge_cover (market orders allowed for this only).
- Cancel every working entry order. Keep TP limits and existing hedge stops.
- Do NOT close losing legs to get flat — hedge them. Do not open new directional risk.

Rules:
- Identify support and resistance from the charts (15m execution, 1H and Daily context).
- Prefer LIMIT entries at support (BUY) / resistance (SELL) with LIMIT take-profit + reverse-side hedge_cover STOP in the same decision.
- hedge_cover opens an opposing hedge leg (force open) — never a closing stop-loss.
- Do not use market orders unless explicitly told or flatten_now is true.
- Close winning legs only; keep protection on losing legs until profitable exit.
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
     "order_id": "optional", "position_id": "for open legs; omit when bracketing a new entry", "reason": "string"}
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
    market_clock: dict[str, Any] | None = None,
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

    clock = dict(market_clock or {})
    if clock.get("flatten_now"):
        net = float(clock.get("net_exposure") or 0)
        instructions = [
            "FLATTEN WINDOW ACTIVE: make the book directionally flat NOW before the gap.",
            f"Net exposure is {net:+.4g}. If non-zero, market_open the opposite side "
            f"with size={abs(net)} and purpose=hedge_cover.",
            "Cancel every working entry order. Keep TP limits. Do not close losing legs.",
            "Do not open new directional entries.",
            f"Respect max_open_positions={max_open_positions} except for this protective hedge.",
        ]

    payload: dict[str, Any] = {
        "phase": phase,
        "order_size": order_size,
        "max_open_positions": max_open_positions,
        "snapshot": snapshot.to_dict(),
        "instructions": " ".join(instructions),
    }
    if clock:
        payload["market_clock"] = clock
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

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        tenant_id: int | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = max(0.0, min(1.0, float(temperature)))
        self.tenant_id = tenant_id
        self.session_factory = session_factory
        self.last_error: str | None = None

    def _record_usage(self, response: object) -> None:
        if self.tenant_id is None or self.session_factory is None:
            return
        session: Session | None = None
        try:
            usage = usage_from_response(response)
            session = self.session_factory()
            UsageRecorderService(SqlAlchemyApiUsageRepository(session)).record(
                self.tenant_id,
                "cac40",
                self.model,
                usage,
            )
            session.commit()
        except Exception:
            logger.exception(
                "Failed to record CAC40 Gemini usage tenant_id=%s model=%s",
                self.tenant_id,
                self.model,
            )
            if session is not None:
                try:
                    session.rollback()
                except Exception:
                    pass
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

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
        market_clock: dict[str, Any] | None = None,
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
                    market_clock=market_clock,
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
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                ),
            )
            self._record_usage(response)
            text = getattr(response, "text", None) or ""
            if not text and getattr(response, "candidates", None):
                text = response.candidates[0].content.parts[0].text  # type: ignore[index]
            return parse_llm_json(text)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("LLM decision failed (%s)", self.model)
            return None
