from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from evenor.adapters.channels.whatsapp_meta import send_whatsapp_text
from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.order_repository import SqlAlchemyOrderRepository
from evenor.adapters.system_clock import SystemClock
from evenor.automation.handlers.order_service import OrderService
from evenor.config.settings import get_settings
from evenor.domain.models.connector import ConnectorDirection, ConnectorType
from evenor.domain.models.hook import HookEvent
from evenor.domain.models.message import ChatMessage, MessageRole
from evenor.domain.models.order import OrderAction, OrderCommand, OrderItem

logger = logging.getLogger(__name__)


class _WhatsAppNotifier:
    def __init__(self, *, phone_number_id: str, access_token: str, admin_wa_id: str) -> None:
        self._phone_number_id = phone_number_id
        self._access_token = access_token
        self._admin_wa_id = admin_wa_id

    def notify_order_event(self, *, action, order, message: str) -> None:
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
            logger.exception("Failed to send admin WhatsApp notification")


class _NullNotifier:
    def notify_order_event(self, *, action, order, message: str) -> None:
        _ = action
        _ = order
        _ = message


def _notifier_for_tenant(session: Session, tenant_id: int):
    conn_repo = SqlAlchemyConnectorRepository(session)
    for direction in (ConnectorDirection.OUT, ConnectorDirection.IN):
        c = conn_repo.find_active(tenant_id, direction=direction, type=ConnectorType.WHATSAPP)
        if c and c.config:
            cfg = c.config
            return _WhatsAppNotifier(
                phone_number_id=str(cfg.get("phone_number_id", "")),
                access_token=str(cfg.get("access_token", "")),
                admin_wa_id=str(cfg.get("admin_wa_id", "")),
            )
    return _NullNotifier()


def _parse_order_command(payload: dict) -> OrderCommand | None:
    action_raw = payload.get("action")
    if not action_raw and isinstance(payload.get("type"), str):
        parts = payload["type"].split(".")
        action_raw = parts[-1] if parts else None
    try:
        action = OrderAction(str(action_raw).strip().lower())
    except (TypeError, ValueError):
        return None
    products = payload.get("products")
    items: list[OrderItem] = []
    if isinstance(products, list):
        for raw in products:
            if not isinstance(raw, dict):
                continue
            try:
                qty = int(raw.get("qty", 0))
            except (TypeError, ValueError):
                continue
            product = str(raw.get("product", "")).strip()
            if qty > 0 and product:
                items.append(OrderItem(qty=qty, product=product))
    return OrderCommand(
        action=action,
        name=str(payload.get("name")).strip() if payload.get("name") else None,
        tel=str(payload.get("tel")).strip() if payload.get("tel") else None,
        address=str(payload.get("address")).strip() if payload.get("address") else None,
        pin=str(payload.get("pin")).strip() if payload.get("pin") else None,
        products=tuple(items),
        reason=str(payload.get("reason")).strip() if payload.get("reason") else None,
        raw_payload=payload,
    )


def handle_order_hook(session: Session, hook: HookEvent) -> None:
    payload = json.loads(hook.payload_json)
    if not isinstance(payload, dict):
        raise ValueError("hook payload must be object")
    command = _parse_order_command(payload)
    if command is None:
        raise ValueError(f"unsupported order action in payload: {payload}")
    settings = get_settings()
    repo = SqlAlchemyOrderRepository(session, hook.tenant_id)
    svc = OrderService(
        repository=repo,
        notifier=_notifier_for_tenant(session, hook.tenant_id),
        clock=SystemClock(),
        modification_window_hours=settings.order_modification_window_hours,
    )
    svc.append_command(
        session_id=hook.session_id,
        command=command,
        command_json=hook.payload_json,
        conversation_context=_load_context(session, hook),
    )


def _load_context(session: Session, hook: HookEvent) -> list[ChatMessage]:
    from evenor.adapters.persistence.orm import MessageRow
    from sqlalchemy import desc, select

    rows = list(
        session.scalars(
            select(MessageRow)
            .where(
                MessageRow.tenant_id == hook.tenant_id,
                MessageRow.session_id == hook.session_id,
            )
            .order_by(desc(MessageRow.id))
            .limit(6)
        )
    )
    rows.reverse()
    out: list[ChatMessage] = []
    for r in rows:
        try:
            role = MessageRole(r.role)
        except ValueError:
            role = MessageRole.USER
        out.append(ChatMessage(role=role, content=r.content))
    return out


def dispatch_hook(session: Session, hook: HookEvent) -> None:
    from evenor.automation.modules.registry import dispatch_hook as registry_dispatch

    registry_dispatch(session, hook)
