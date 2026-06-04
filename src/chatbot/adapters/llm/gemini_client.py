from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types

from chatbot.config.settings import Settings, get_settings
from chatbot.domain.contracts.llm_client import LlmResult, LlmUsage
from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage, MessageRole


class GeminiLlmClient:
    """Reads model id and API key from ``get_settings()`` on each call so `.env` edits apply without restart."""

    def __init__(self, *, model_attr: Literal["chat_model", "rewrite_model"]) -> None:
        self._model_attr = model_attr
        self._client: genai.Client | None = None
        self._client_api_key: str | None = None

    def _client_and_model(self) -> tuple[genai.Client, str]:
        s: Settings = get_settings()
        key = s.gemini_api_key or ""
        if self._client is None or key != self._client_api_key:
            self._client = genai.Client(api_key=key) if key else genai.Client()
            self._client_api_key = key
        model = getattr(s, self._model_attr)
        return self._client, model

    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
        attachments: list[Attachment] | None = None,
    ) -> LlmResult:
        client, model = self._client_and_model()
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.role == MessageRole.USER),
            default=-1,
        )
        contents: list[types.Content] = []
        for i, m in enumerate(messages):
            if m.role == MessageRole.SYSTEM:
                continue
            role = "user" if m.role == MessageRole.USER else "model"
            parts: list[types.Part] = [types.Part.from_text(text=m.content)]
            if (
                m.role == MessageRole.USER
                and i == last_user_idx
                and attachments
            ):
                for att in attachments:
                    parts.append(
                        types.Part.from_bytes(data=att.data, mime_type=att.mime_type)
                    )
            contents.append(types.Content(role=role, parts=parts))
        response = client.models.generate_content(
            model=model,
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
