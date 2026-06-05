from __future__ import annotations

import httpx
import pytest

from chatbot.automation.handlers.admin_notifier import WhatsAppAdminNotifier
from chatbot.config.settings import Settings
from chatbot.domain.models.order import OrderAction


def test_whatsapp_admin_notifier_swallows_graph_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph API 401 (bad/expired token) must not break order persistence or chat flow."""

    def _boom(**_kwargs) -> None:
        req = httpx.Request("POST", "https://graph.facebook.com/v21.0/phone/messages")
        resp = httpx.Response(401, request=req, json={"error": {"message": "Invalid OAuth access token"}})
        raise httpx.HTTPStatusError("401", request=req, response=resp)

    monkeypatch.setattr("chatbot.automation.handlers.admin_notifier.send_whatsapp_text", _boom)

    notifier = WhatsAppAdminNotifier(
        phone_number_id="209262928928431",
        access_token="bad-token",
        admin_wa_id="23000000000",
    )
    # Must not raise (matches production: log and continue).
    notifier.notify_order_event(
        action=OrderAction.CREATE,
        order=None,
        message="[Order created] id=1",
    )


@pytest.mark.integration
def test_whatsapp_admin_notifier_sends_real_message() -> None:
    """Live integration test — skipped automatically when credentials are absent.

    Run manually to verify the admin WhatsApp number receives a notification:
        pytest -m integration -v
    """
    s = Settings()
    missing = [
        name
        for name, val in [
            ("WHATSAPP_PHONE_NUMBER_ID", s.whatsapp_phone_number_id),
            ("WHATSAPP_ACCESS_TOKEN", s.whatsapp_access_token),
            ("WHATSAPP_ADMIN_WA_ID", s.whatsapp_admin_wa_id),
        ]
        if not val.strip()
    ]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")

    notifier = WhatsAppAdminNotifier(
        phone_number_id=s.whatsapp_phone_number_id,
        access_token=s.whatsapp_access_token,
        admin_wa_id=s.whatsapp_admin_wa_id,
    )
    # This actually calls the Graph API and delivers a message to WHATSAPP_ADMIN_WA_ID.
    notifier.notify_order_event(
        action=OrderAction.CREATE,
        order=None,
        message=(
            "[Test] Admin notifier integration test\n"
            f"Admin WA ID: {s.whatsapp_admin_wa_id}\n"
            f"Phone number ID: {s.whatsapp_phone_number_id}"
        ),
    )
