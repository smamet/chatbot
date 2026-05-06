from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from chatbot.adapters.channels.whatsapp_meta import send_whatsapp_text
from chatbot.domain.models.order import OrderAction, OrderSnapshot

log = logging.getLogger(__name__)


@runtime_checkable
class AdminNotifier(Protocol):
    def notify_order_event(self, *, action: OrderAction, order: OrderSnapshot | None, message: str) -> None:
        ...


class NullAdminNotifier:
    def notify_order_event(self, *, action: OrderAction, order: OrderSnapshot | None, message: str) -> None:
        _ = action
        _ = order
        _ = message


class WhatsAppAdminNotifier:
    def __init__(
        self,
        *,
        phone_number_id: str,
        access_token: str,
        admin_wa_id: str,
    ) -> None:
        self._phone_number_id = phone_number_id.strip()
        self._access_token = access_token.strip()
        self._admin_wa_id = admin_wa_id.strip()

    def notify_order_event(self, *, action: OrderAction, order: OrderSnapshot | None, message: str) -> None:
        _ = action
        _ = order
        if not (self._phone_number_id and self._access_token and self._admin_wa_id):
            return
        try:
            send_whatsapp_text(
                phone_number_id=self._phone_number_id,
                access_token=self._access_token,
                to_wa_id=self._admin_wa_id,
                text=message,
            )
        except Exception:
            log.exception("Failed to send admin WhatsApp notification")
