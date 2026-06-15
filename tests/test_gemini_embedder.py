from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.genai import errors

from chatbot.adapters.embeddings import gemini_embedder as embedder_mod
from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from tests.conftest import TestSettings


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeResponse:
    def __init__(self, values: list[list[float]]) -> None:
        self.embeddings = [_FakeEmbedding(v) for v in values]


def _test_settings(**overrides) -> TestSettings:
    from cryptography.fernet import Fernet

    base = {
        "gemini_api_key": "test-key",
        "admin_token": "admin",
        "app_secret_key": Fernet.generate_key().decode(),
        "session_secret": "session",
    }
    base.update(overrides)
    return TestSettings(**base)


def _stub_client(
    monkeypatch: pytest.MonkeyPatch, embedder: GeminiEmbedder, client: MagicMock
) -> None:
    embedder._client = client
    embedder._client_api_key = "test-key"
    monkeypatch.setattr(
        embedder,
        "_client_and_model",
        lambda: (client, "gemini-embedding-001"),
    )


def test_embed_texts_retries_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(embed_retry_max=5, embed_retry_base_429_seconds=30.0)
    monkeypatch.setattr(embedder_mod, "get_settings", lambda: settings)
    sleeps: list[float] = []
    monkeypatch.setattr(embedder_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    calls = {"count": 0}
    client = MagicMock()

    def embed_side_effect(*, model, contents):
        calls["count"] += 1
        if calls["count"] == 1:
            response = MagicMock()
            response.headers = {"Retry-After": "0.01"}
            raise errors.APIError(
                429,
                {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
                response=response,
            )
        return _FakeResponse([[0.1, 0.2, 0.3]])

    client.models.embed_content.side_effect = embed_side_effect

    embedder = GeminiEmbedder(api_key="test-key")
    _stub_client(monkeypatch, embedder, client)

    result = embedder.embed_texts(["hello"])
    assert result == [[0.1, 0.2, 0.3]]
    assert calls["count"] == 2
    assert sleeps == [0.01]


def test_embed_texts_retries_503_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(
        embed_retry_max=3,
        embed_retry_base_503_seconds=5.0,
    )
    monkeypatch.setattr(embedder_mod, "get_settings", lambda: settings)
    sleeps: list[float] = []
    monkeypatch.setattr(embedder_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    calls = {"count": 0}
    client = MagicMock()

    def embed_side_effect(*, model, contents):
        calls["count"] += 1
        if calls["count"] < 3:
            raise errors.APIError(
                503,
                {"error": {"code": 503, "status": "UNAVAILABLE"}},
                response=MagicMock(headers={}),
            )
        return _FakeResponse([[1.0, 2.0]])

    client.models.embed_content.side_effect = embed_side_effect

    embedder = GeminiEmbedder(api_key="test-key")
    _stub_client(monkeypatch, embedder, client)

    result = embedder.embed_texts(["world"])
    assert result == [[1.0, 2.0]]
    assert calls["count"] == 3
    assert sleeps == [5.0, 10.0]


def test_embed_texts_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings(embed_retry_max=2)
    monkeypatch.setattr(embedder_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(embedder_mod.time, "sleep", lambda _seconds: None)

    client = MagicMock()
    client.models.embed_content.side_effect = errors.APIError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
        response=MagicMock(headers={}),
    )

    embedder = GeminiEmbedder(api_key="test-key")
    _stub_client(monkeypatch, embedder, client)

    with pytest.raises(errors.APIError):
        embedder.embed_texts(["fail"])
    assert client.models.embed_content.call_count == 2
