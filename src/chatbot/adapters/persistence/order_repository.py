from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import OrderEventRow, OrderItemRow, OrderRow
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.domain.models.order import OrderAction, OrderCommand, OrderEvent, OrderItem, OrderSnapshot, OrderStatus


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _context_to_json(messages: list[ChatMessage]) -> str:
    payload = [{"role": m.role.value, "content": m.content} for m in messages]
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _context_from_json(payload: str) -> tuple[ChatMessage, ...]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return ()
    out: list[ChatMessage] = []
    if not isinstance(parsed, list):
        return ()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role_raw = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        try:
            role = MessageRole(role_raw)
        except ValueError:
            role = MessageRole.USER
        out.append(ChatMessage(role=role, content=content))
    return tuple(out)


def _to_order_snapshot(row: OrderRow) -> OrderSnapshot:
    try:
        status = OrderStatus(row.status)
    except ValueError:
        status = OrderStatus.PENDING
    items = tuple(OrderItem(qty=i.qty, product=i.product) for i in row.items)
    return OrderSnapshot(
        id=row.id,
        session_id=row.session_id,
        customer_key=row.customer_key,
        customer_name=row.customer_name,
        customer_tel=row.customer_tel,
        delivery_address=row.delivery_address,
        delivery_pin=row.delivery_pin,
        status=status,
        editable_until=_as_utc(row.editable_until),
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        deleted_at=_as_utc(row.deleted_at) if row.deleted_at else None,
        items=items,
    )


class SqlAlchemyOrderRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_order(
        self,
        *,
        session_id: str,
        customer_key: str,
        command: OrderCommand,
        editable_until: datetime,
    ) -> OrderSnapshot:
        now = _as_utc(datetime.now(UTC))
        row = OrderRow(
            session_id=session_id,
            customer_key=customer_key,
            customer_name=command.name,
            customer_tel=command.tel,
            delivery_address=command.address,
            delivery_pin=command.pin,
            status=OrderStatus.PENDING.value,
            editable_until=_as_utc(editable_until),
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        for item in command.products:
            row.items.append(OrderItemRow(qty=item.qty, product=item.product, created_at=now))
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _to_order_snapshot(row)

    def find_latest_editable_order(self, *, customer_key: str, now: datetime) -> OrderSnapshot | None:
        stmt = (
            select(OrderRow)
            .where(
                OrderRow.customer_key == customer_key,
                OrderRow.status == OrderStatus.PENDING.value,
                OrderRow.editable_until >= _as_utc(now),
            )
            .order_by(desc(OrderRow.id))
            .limit(1)
        )
        row = self._session.scalar(stmt)
        return _to_order_snapshot(row) if row else None

    def find_latest_order(self, *, customer_key: str) -> OrderSnapshot | None:
        stmt = (
            select(OrderRow)
            .where(OrderRow.customer_key == customer_key)
            .order_by(desc(OrderRow.id))
            .limit(1)
        )
        row = self._session.scalar(stmt)
        return _to_order_snapshot(row) if row else None

    def update_order(self, *, order_id: int, command: OrderCommand, updated_at: datetime) -> OrderSnapshot:
        row = self._session.get(OrderRow, order_id)
        if row is None:
            raise ValueError(f"Order {order_id} not found")
        if command.name:
            row.customer_name = command.name
        if command.tel:
            row.customer_tel = command.tel
        if command.address:
            row.delivery_address = command.address
        if command.pin:
            row.delivery_pin = command.pin
        row.updated_at = _as_utc(updated_at)
        row.items.clear()
        for item in command.products:
            row.items.append(OrderItemRow(qty=item.qty, product=item.product, created_at=row.updated_at))
        self._session.flush()
        self._session.refresh(row)
        return _to_order_snapshot(row)

    def delete_order(self, *, order_id: int, deleted_at: datetime) -> OrderSnapshot:
        row = self._session.get(OrderRow, order_id)
        if row is None:
            raise ValueError(f"Order {order_id} not found")
        ts = _as_utc(deleted_at)
        row.status = OrderStatus.DELETED.value
        row.deleted_at = ts
        row.updated_at = ts
        self._session.flush()
        self._session.refresh(row)
        return _to_order_snapshot(row)

    def append_event(
        self,
        *,
        order_id: int | None,
        session_id: str,
        customer_key: str,
        action: OrderAction,
        result: str,
        command_json: str,
        conversation_context: list[ChatMessage],
        created_at: datetime,
        error_detail: str | None = None,
    ) -> OrderEvent:
        row = OrderEventRow(
            order_id=order_id,
            session_id=session_id,
            customer_key=customer_key,
            action=action.value,
            result=result,
            command_json=command_json,
            conversation_context=_context_to_json(conversation_context),
            error_detail=error_detail,
            created_at=_as_utc(created_at),
        )
        self._session.add(row)
        self._session.flush()
        return OrderEvent(
            order_id=row.order_id,
            session_id=row.session_id,
            customer_key=row.customer_key,
            action=OrderAction(row.action),
            result=row.result,
            command_json=row.command_json,
            conversation_context=_context_from_json(row.conversation_context),
            created_at=_as_utc(row.created_at),
            error_detail=row.error_detail,
        )

    def list_ready_orders(self, *, now: datetime, limit: int = 100) -> list[OrderSnapshot]:
        stmt = (
            select(OrderRow)
            .where(
                OrderRow.status == OrderStatus.PENDING.value,
                OrderRow.editable_until <= _as_utc(now),
            )
            .order_by(OrderRow.editable_until.asc(), OrderRow.id.asc())
            .limit(limit)
        )
        rows = list(self._session.scalars(stmt))
        return [_to_order_snapshot(r) for r in rows]
