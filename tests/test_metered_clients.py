from __future__ import annotations

from unittest.mock import MagicMock

from chatbot.adapters.embeddings.metered_embedder import MeteredEmbedder
from chatbot.adapters.llm.metered_llm_client import MeteredLlmClient
from chatbot.application.usage_recorder_service import UsageRecorderService
from chatbot.domain.contracts.llm_client import LlmResult, LlmUsage
from chatbot.domain.models.message import ChatMessage, MessageRole


class _FakeLlm:
    def generate_chat(self, *, system_instruction, messages, attachments=None):
        _ = system_instruction, messages, attachments
        return LlmResult(text="ok", usage=LlmUsage(prompt_tokens=4, candidates_tokens=2, total_tokens=6))


def test_metered_llm_records_usage() -> None:
    recorder = MagicMock(spec=UsageRecorderService)
    client = MeteredLlmClient(
        inner=_FakeLlm(),
        tenant_id=7,
        operation="chat",
        model="gemini-2.0-flash",
        recorder=recorder,
    )
    out = client.generate_chat(
        system_instruction="sys",
        messages=[ChatMessage(role=MessageRole.USER, content="hi")],
    )
    assert out.text == "ok"
    recorder.record.assert_called_once()
    assert recorder.record.call_args.args[:3] == (7, "chat", "gemini-2.0-flash")


def test_metered_embedder_records_usage() -> None:
    inner = MagicMock()
    inner.embed_texts.return_value = [[0.1]]
    inner.last_embed_meta = ("text-embedding-004", LlmUsage(prompt_tokens=3))
    recorder = MagicMock(spec=UsageRecorderService)
    embedder = MeteredEmbedder(
        inner=inner,
        tenant_id=3,
        operation="embed_ingest",
        model="text-embedding-004",
        recorder=recorder,
    )
    vectors = embedder.embed_texts(["hello"])
    assert vectors == [[0.1]]
    recorder.record.assert_called_once_with(3, "embed_ingest", "text-embedding-004", LlmUsage(prompt_tokens=3))
