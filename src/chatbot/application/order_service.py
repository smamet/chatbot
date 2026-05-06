from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta

from chatbot.application.admin_notifier import AdminNotifier
from chatbot.domain.contracts.clock import Clock
from chatbot.domain.contracts.order_repository import OrderRepository
from chatbot.domain.models.message import ChatMessage
from chatbot.domain.models.order import OrderAction, OrderCommand, OrderSnapshot


@dataclass(frozen=True, slots=True)
class OrderServiceResult:
    action: OrderAction
    result: str
    order: OrderSnapshot | None
    message: str


class OrderService:
    def __init__(
        self,
        *,
        repository: OrderRepository,
        notifier: AdminNotifier,
        clock: Clock,
        modification_window_hours: int = 6,
    ) -> None:
        self._repository = repository
        self._notifier = notifier
        self._clock = clock
        self._modification_window_hours = max(1, modification_window_hours)

    def append_command(
        self,
        *,
        session_id: str,
        command: OrderCommand,
        command_json: str | None,
        conversation_context: list[ChatMessage],
    ) -> OrderServiceResult:
        now = self._clock.now()
        customer_key = self._customer_key(session_id=session_id, tel=command.tel)
        payload_json = self._command_json(command=command, command_json=command_json)

        if command.action is OrderAction.CREATE:
            editable_until = now + timedelta(hours=self._modification_window_hours)
            order = self._repository.create_order(
                session_id=session_id,
                customer_key=customer_key,
                command=command,
                editable_until=editable_until,
            )
            message = self._format_message("created", order, command.reason)
            self._repository.append_event(
                order_id=order.id,
                session_id=session_id,
                customer_key=customer_key,
                action=command.action,
                result="created",
                command_json=payload_json,
                conversation_context=conversation_context,
                created_at=now,
            )
            self._notifier.notify_order_event(action=command.action, order=order, message=message)
            return OrderServiceResult(action=command.action, result="created", order=order, message=message)

        latest = self._repository.find_latest_editable_order(customer_key=customer_key, now=now)
        if latest is None:
            err = "No editable order found in modification window"
            self._repository.append_event(
                order_id=None,
                session_id=session_id,
                customer_key=customer_key,
                action=command.action,
                result="rejected",
                command_json=payload_json,
                conversation_context=conversation_context,
                created_at=now,
                error_detail=err,
            )
            message = f"[Order {command.action.value} rejected] customer={customer_key} reason={err}"
            self._notifier.notify_order_event(action=command.action, order=None, message=message)
            return OrderServiceResult(action=command.action, result="rejected", order=None, message=message)

        if command.action is OrderAction.UPDATE:
            order = self._repository.update_order(order_id=latest.id, command=command, updated_at=now)
            result = "updated"
        else:
            order = self._repository.delete_order(order_id=latest.id, deleted_at=now)
            result = "deleted"
        message = self._format_message(result, order, command.reason)
        self._repository.append_event(
            order_id=order.id,
            session_id=session_id,
            customer_key=customer_key,
            action=command.action,
            result=result,
            command_json=payload_json,
            conversation_context=conversation_context,
            created_at=now,
        )
        self._notifier.notify_order_event(action=command.action, order=order, message=message)
        return OrderServiceResult(action=command.action, result=result, order=order, message=message)

    def list_ready_orders(self, *, limit: int = 100) -> list[OrderSnapshot]:
        return self._repository.list_ready_orders(now=self._clock.now(), limit=limit)

    @staticmethod
    def _customer_key(*, session_id: str, tel: str | None) -> str:
        raw = (tel or "").strip()
        if raw:
            return "".join(ch for ch in raw if ch.isdigit()) or raw
        return session_id.strip()

    @staticmethod
    def _command_json(*, command: OrderCommand, command_json: str | None) -> str:
        if command_json:
            return command_json
        payload = command.raw_payload or {"action": command.action.value}
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _format_message(result: str, order: OrderSnapshot | None, reason: str | None) -> str:
        if order is None:
            return f"[Order {result}]"
        products = ", ".join(f"{item.qty}x {item.product}" for item in order.items) or "n/a"
        reason_part = f"\nReason: {reason}" if reason else ""
        return (
            f"[Order {result}] id={order.id}"
            f"\nCustomer: {order.customer_name or 'n/a'} ({order.customer_tel or order.customer_key})"
            f"\nAddress: {order.delivery_address or 'n/a'}"
            f"\nPin: {order.delivery_pin or 'n/a'}"
            f"\nProducts: {products}"
            f"\nEditable until: {order.editable_until.isoformat()}"
            f"{reason_part}"
        )
