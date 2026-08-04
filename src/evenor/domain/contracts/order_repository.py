from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from evenor.domain.models.message import ChatMessage
from evenor.domain.models.order import OrderAction, OrderCommand, OrderEvent, OrderSnapshot


@runtime_checkable
class OrderRepository(Protocol):
    def create_order(
        self,
        *,
        session_id: str,
        customer_key: str,
        command: OrderCommand,
        editable_until: datetime,
    ) -> OrderSnapshot:
        ...

    def find_latest_editable_order(
        self,
        *,
        customer_key: str,
        now: datetime,
    ) -> OrderSnapshot | None:
        ...

    def find_latest_order(self, *, customer_key: str) -> OrderSnapshot | None:
        ...

    def update_order(
        self,
        *,
        order_id: int,
        command: OrderCommand,
        updated_at: datetime,
    ) -> OrderSnapshot:
        ...

    def delete_order(self, *, order_id: int, deleted_at: datetime) -> OrderSnapshot:
        ...

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
        ...

    def list_ready_orders(self, *, now: datetime, limit: int = 100) -> list[OrderSnapshot]:
        ...
