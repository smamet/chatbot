from __future__ import annotations

from google import genai
from google.genai import types

from chatbot.domain.contracts.llm_client import LlmResult, LlmUsage
from chatbot.domain.models.message import ChatMessage, MessageRole


class GeminiLlmClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
    ) -> LlmResult:
        contents: list[types.Content] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                continue
            role = "user" if m.role == MessageRole.USER else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m.content)],
                )
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
        text = (response.text or "").strip()
        usage = _usage_from_response(response)
        return LlmResult(text=text, usage=usage)


def _usage_from_response(response: object) -> LlmUsage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return LlmUsage()
    return LlmUsage(
        prompt_tokens=getattr(meta, "prompt_token_count", None),
        candidates_tokens=getattr(meta, "candidates_token_count", None),
        total_tokens=getattr(meta, "total_token_count", None),
    )
