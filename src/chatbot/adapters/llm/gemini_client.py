from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types

from chatbot.adapters.gemini_usage import usage_from_response
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.contracts.llm_client import LlmResult
from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage, MessageRole


class GeminiLlmClient:
    """Gemini client with optional per-request API key and model override."""

    def __init__(
        self,
        *,
        model_attr: Literal["chat_model", "rewrite_model"] | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model_attr = model_attr
        self._fixed_model = model
        self._api_key = api_key
        self._client: genai.Client | None = None
        self._client_api_key: str | None = None

    def _client_and_model(self) -> tuple[genai.Client, str]:
        s: Settings = get_settings()
        key = (self._api_key or s.gemini_api_key or "").strip()
        if self._client is None or key != self._client_api_key:
            self._client = genai.Client(api_key=key) if key else genai.Client()
            self._client_api_key = key
        if self._fixed_model:
            return self._client, self._fixed_model
        assert self._model_attr is not None
        return self._client, getattr(s, self._model_attr)

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
            if m.role == MessageRole.USER and i == last_user_idx and attachments:
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
        return LlmResult(text=text, usage=usage_from_response(response))
