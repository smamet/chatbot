from __future__ import annotations

from datetime import UTC, datetime, timedelta

from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.order_repository import SqlAlchemyOrderRepository
from evenor.adapters.persistence.orm import OrderEventRow
from evenor.automation.handlers.order_service import OrderService
from evenor.domain.models.message import ChatMessage, MessageRole
from evenor.domain.models.order import OrderAction, OrderCommand, OrderItem


class _Clock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def set(self, now: datetime) -> None:
        self._now = now


class _Notifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify_order_event(self, *, action, order, message: str) -> None:
        _ = action
        _ = order
        self.messages.append(message)


def _context(n: int = 6) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for i in range(n):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        out.append(ChatMessage(role=role, content=f"m-{i}"))
    return out


def test_order_service_create_persists_order_and_event(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    now = datetime(2026, 5, 5, 9, 0, tzinfo=UTC)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyOrderRepository(session, tenant.id)
        notifier = _Notifier()
        svc = OrderService(repository=repo, notifier=notifier, clock=_Clock(now), modification_window_hours=6)
        cmd = OrderCommand(
            action=OrderAction.CREATE,
            name="Ana",
            tel="23057770000",
            address="Quatre Bornes",
            products=(OrderItem(qty=2, product="Diffuser"),),
            raw_payload={"action": "create"},
        )
        result = svc.append_command(
            session_id="whatsapp:23057770000",
            command=cmd,
            command_json='{"action":"create"}',
            conversation_context=_context(6),
        )
        session.commit()
        ready = svc.list_ready_orders(limit=10)
    finally:
        session.close()
    assert result.result == "created"
    assert result.order is not None
    assert result.order.customer_name == "Ana"
    assert result.order.editable_until == now + timedelta(hours=6)
    assert ready == []
    assert notifier.messages


def test_order_service_update_within_window_updates_latest_order(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    base = datetime(2026, 5, 5, 9, 0, tzinfo=UTC)
    clock = _Clock(base)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyOrderRepository(session, tenant.id)
        svc = OrderService(repository=repo, notifier=_Notifier(), clock=clock, modification_window_hours=6)
        svc.append_command(
            session_id="whatsapp:1",
            command=OrderCommand(
                action=OrderAction.CREATE,
                tel="2301",
                address="A1",
                products=(OrderItem(qty=1, product="Old"),),
            ),
            command_json='{"action":"create"}',
            conversation_context=_context(),
        )
        clock.set(base + timedelta(hours=1))
        out = svc.append_command(
            session_id="whatsapp:1",
            command=OrderCommand(
                action=OrderAction.UPDATE,
                tel="2301",
                address="A2",
                products=(OrderItem(qty=3, product="New"),),
            ),
            command_json='{"action":"update"}',
            conversation_context=_context(),
        )
        session.commit()
    finally:
        session.close()
    assert out.result == "updated"
    assert out.order is not None
    assert out.order.delivery_address == "A2"
    assert out.order.items[0].qty == 3


def test_order_service_create_same_customer_creates_second_order(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    now = datetime(2026, 5, 5, 9, 0, tzinfo=UTC)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyOrderRepository(session, tenant.id)
        svc = OrderService(repository=repo, notifier=_Notifier(), clock=_Clock(now), modification_window_hours=6)
        first = svc.append_command(
            session_id="s1",
            command=OrderCommand(action=OrderAction.CREATE, tel="2305", products=(OrderItem(qty=1, product="A"),)),
            command_json='{"action":"create"}',
            conversation_context=_context(),
        )
        second = svc.append_command(
            session_id="s1",
            command=OrderCommand(action=OrderAction.CREATE, tel="2305", products=(OrderItem(qty=1, product="B"),)),
            command_json='{"action":"create"}',
            conversation_context=_context(),
        )
        session.commit()
    finally:
        session.close()
    assert first.order is not None and second.order is not None
    assert first.order.id != second.order.id


def test_order_service_delete_marks_deleted_and_logs_context(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    now = datetime(2026, 5, 5, 9, 0, tzinfo=UTC)
    clock = _Clock(now)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyOrderRepository(session, tenant.id)
        svc = OrderService(repository=repo, notifier=_Notifier(), clock=clock, modification_window_hours=6)
        svc.append_command(
            session_id="s-del",
            command=OrderCommand(action=OrderAction.CREATE, tel="2307", products=(OrderItem(qty=1, product="X"),)),
            command_json='{"action":"create"}',
            conversation_context=_context(),
        )
        out = svc.append_command(
            session_id="s-del",
            command=OrderCommand(action=OrderAction.DELETE, tel="2307", reason="customer cancelled"),
            command_json='{"action":"delete"}',
            conversation_context=_context(6),
        )
        session.flush()
        events = list(session.query(OrderEventRow).order_by(OrderEventRow.id.asc()))
        session.commit()
    finally:
        session.close()
    assert out.result == "deleted"
    assert out.order is not None
    assert out.order.status.value == "deleted"
    assert len(events) == 2
    assert '"role":"user"' in events[-1].conversation_context


def test_order_service_expired_update_is_rejected_and_logged(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    base = datetime(2026, 5, 5, 9, 0, tzinfo=UTC)
    clock = _Clock(base)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyOrderRepository(session, tenant.id)
        svc = OrderService(repository=repo, notifier=_Notifier(), clock=clock, modification_window_hours=6)
        svc.append_command(
            session_id="s-exp",
            command=OrderCommand(action=OrderAction.CREATE, tel="2309", products=(OrderItem(qty=1, product="X"),)),
            command_json='{"action":"create"}',
            conversation_context=_context(),
        )
        clock.set(base + timedelta(hours=7))
        out = svc.append_command(
            session_id="s-exp",
            command=OrderCommand(action=OrderAction.UPDATE, tel="2309", address="late"),
            command_json='{"action":"update"}',
            conversation_context=_context(),
        )
        events = list(session.query(OrderEventRow).order_by(OrderEventRow.id.asc()))
        session.commit()
    finally:
        session.close()
    assert out.result == "rejected"
    assert events[-1].result == "rejected"
    assert events[-1].error_detail is not None


def test_order_service_list_ready_orders_returns_past_window(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    base = datetime(2026, 5, 5, 9, 0, tzinfo=UTC)
    clock = _Clock(base)
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyOrderRepository(session, tenant.id)
        svc = OrderService(repository=repo, notifier=_Notifier(), clock=clock, modification_window_hours=6)
        svc.append_command(
            session_id="s-ready",
            command=OrderCommand(action=OrderAction.CREATE, tel="2310", products=(OrderItem(qty=1, product="R"),)),
            command_json='{"action":"create"}',
            conversation_context=_context(),
        )
        clock.set(base + timedelta(hours=7))
        ready = svc.list_ready_orders(limit=10)
        session.commit()
    finally:
        session.close()
    assert len(ready) == 1
