from __future__ import annotations

import json

from evenor.adapters.channels import instagram_meta, messenger_meta, whatsapp_meta


class _Resp:
    def raise_for_status(self) -> None:
        return None


class _DummyClient:
    last_content: str | None = None

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return None

    def post(self, url, *, headers, content):  # type: ignore[no-untyped-def]
        _DummyClient.last_content = content
        return _Resp()


def test_send_messenger_text_uses_plain_text_formatting(monkeypatch) -> None:
    monkeypatch.setattr(messenger_meta.httpx, "Client", _DummyClient)
    messenger_meta.send_messenger_text(
        page_access_token="t",
        recipient_psid="psid",
        text="# Title\n- **Room**",
    )
    payload = json.loads(_DummyClient.last_content or "{}")
    assert payload["message"]["text"] == "Title\n• Room"


def test_send_instagram_text_uses_plain_text_formatting(monkeypatch) -> None:
    monkeypatch.setattr(instagram_meta.httpx, "Client", _DummyClient)
    instagram_meta.send_instagram_text(
        ig_user_id="igid",
        access_token="t",
        recipient_igsid="igsid",
        text="## T\n* `A`",
    )
    payload = json.loads(_DummyClient.last_content or "{}")
    assert payload["message"]["text"] == "T\n• A"


def test_send_whatsapp_text_keeps_whatsapp_emphasis(monkeypatch) -> None:
    monkeypatch.setattr(whatsapp_meta.httpx, "Client", _DummyClient)
    whatsapp_meta.send_whatsapp_text(
        phone_number_id="pid",
        access_token="t",
        to_wa_id="waid",
        text="*Room* _Mist_",
    )
    payload = json.loads(_DummyClient.last_content or "{}")
    assert payload["text"]["body"] == "*Room* _Mist_"
