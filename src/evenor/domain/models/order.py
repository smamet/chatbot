from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from evenor.domain.models.message import ChatMessage


class OrderStatus(str, Enum):
    PENDING = "pending"
    DELETED = "deleted"


class OrderAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class OrderItem:
    qty: int
    product: str


@dataclass(frozen=True, slots=True)
class OrderCommand:
    action: OrderAction
    name: str | None = None
    tel: str | None = None
    address: str | None = None
    pin: str | None = None
    products: tuple[OrderItem, ...] = ()
    reason: str | None = None
    raw_payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    id: int
    session_id: str
    customer_key: str
    customer_name: str | None
    customer_tel: str | None
    delivery_address: str | None
    delivery_pin: str | None
    status: OrderStatus
    editable_until: datetime
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    items: tuple[OrderItem, ...]


@dataclass(frozen=True, slots=True)
class OrderEvent:
    order_id: int | None
    session_id: str
    customer_key: str
    action: OrderAction
    result: str
    command_json: str
    conversation_context: tuple[ChatMessage, ...]
    created_at: datetime
    error_detail: str | None = None
