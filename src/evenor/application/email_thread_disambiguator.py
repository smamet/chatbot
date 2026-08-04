from __future__ import annotations

import json
import re
from dataclasses import dataclass

from evenor.domain.contracts.llm_client import LlmClient
from evenor.domain.models.message import ChatMessage, MessageRole

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ThreadDisambiguationResult:
    same_thread: bool
    confidence: float
    thread_key: str | None = None
    llm_called: bool = False
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class EmailThreadDisambiguator:
    def __init__(
        self,
        *,
        llm: LlmClient | None,
        min_confidence: float = 0.7,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._min_confidence = min_confidence
        self._enabled = enabled

    def disambiguate(
        self,
        *,
        inbound_subject: str,
        body_preview: str,
        candidates: list[dict[str, str]],
    ) -> ThreadDisambiguationResult:
        if not self._enabled or self._llm is None or not candidates:
            return ThreadDisambiguationResult(same_thread=False, confidence=0.0)

        lines = [
            "Decide if the inbound email belongs to an existing thread or starts a new one.",
            "Reply with JSON only: {\"same_thread\": true|false, \"confidence\": 0.0-1.0, \"thread_key\": \"...\"|null}",
            f"Inbound normalized subject: {inbound_subject}",
            f"Inbound body preview: {body_preview[:200]}",
            "Candidate threads:",
        ]
        for cand in candidates[:3]:
            lines.append(
                f"- thread_key={cand.get('thread_key','')} subject={cand.get('subject','')} "
                f"last_activity={cand.get('last_activity','')}"
            )
        system = (
            "You classify email threads. If clearly a new topic despite reply headers, "
            "return same_thread=false. thread_key must be one of the candidate keys or null."
        )
        result = self._llm.generate_chat(
            system_instruction=system,
            messages=[ChatMessage(role=MessageRole.USER, content="\n".join(lines))],
        )
        prompt_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.candidates_tokens
        parsed = self._parse_json(result.text)
        same_thread = bool(parsed.get("same_thread"))
        confidence = float(parsed.get("confidence", 0.0))
        thread_key = parsed.get("thread_key")
        if isinstance(thread_key, str):
            thread_key = thread_key.strip() or None
        else:
            thread_key = None
        if confidence < self._min_confidence:
            return ThreadDisambiguationResult(
                same_thread=False,
                confidence=confidence,
                llm_called=True,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        return ThreadDisambiguationResult(
            same_thread=same_thread,
            confidence=confidence,
            thread_key=thread_key,
            llm_called=True,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = (text or "").strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            m = _JSON_BLOCK.search(text)
            if not m:
                return {}
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
