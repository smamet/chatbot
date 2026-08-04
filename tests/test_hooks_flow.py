from __future__ import annotations

import json
import uuid

from evenor.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from evenor.adapters.persistence.engine import create_db_engine, session_factory
from evenor.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from evenor.application.chat_service import ChatService
from evenor.application.hook_extractor import extract_hook
from evenor.automation.handlers import dispatch_hook
from evenor.domain.constants import HOOK_MARKER
from evenor.domain.contracts.llm_client import LlmResult, LlmUsage
from evenor.domain.models.hook import HookStatus
from evenor.domain.models.message import ChatMessage, MessageRole


class FakeLlm:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate_chat(self, *, system_instruction: str, messages, attachments=None) -> LlmResult:
        _ = system_instruction, messages, attachments
        return LlmResult(text=self._reply, usage=LlmUsage())


def test_chat_service_emits_hook_not_orders(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    payload = {"type": "order.create", "action": "create", "tel": "23057770000", "products": [{"qty": 1, "product": "X"}]}
    reply = f"Done.\n{HOOK_MARKER}\n{json.dumps(payload)}"
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        hooks = SqlAlchemyHookEventRepository(session, tenant.id)
        svc = ChatService(
            settings=test_settings,
            tenant=tenant,
            llm=FakeLlm(reply),
            repo=repo,
            rag=None,
            hook_repo=hooks,
        )
        out = svc.handle_user_message(f"s-{uuid.uuid4().hex}", "order please")
        session.commit()
        pending = hooks.list_by_tenant(status=HookStatus.PENDING)
    finally:
        session.close()
    assert out.text == "Done."
    assert len(pending) == 1
    assert pending[0].type == "order.create"


def test_worker_dispatch_order(test_settings, test_tenant) -> None:
    tenant, _ = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        hooks = SqlAlchemyHookEventRepository(session, tenant.id)
        hook = hooks.create(
            session_id="whatsapp:1",
            hook_type="order",
            payload_json=json.dumps(
                {
                    "action": "create",
                    "name": "Ana",
                    "tel": "23057770000",
                    "products": [{"qty": 2, "product": "Diffuser"}],
                }
            ),
        )
        session.commit()
        session2 = factory()
        try:
            global_repo = SqlAlchemyHookEventRepository(session2, tenant_id=None)
            claimed = global_repo.claim_pending(limit=5)
            assert len(claimed) == 1
            dispatch_hook(session2, claimed[0])
            global_repo.update_status(claimed[0].id, status=HookStatus.DONE)
            session2.commit()
        finally:
            session2.close()
        from evenor.adapters.persistence.order_repository import SqlAlchemyOrderRepository

        order = SqlAlchemyOrderRepository(session, tenant.id).find_latest_order(customer_key="23057770000")
    finally:
        session.close()
    assert order is not None
    assert order.items[0].product == "Diffuser"


def test_hook_events_tenant_isolation(test_settings, test_tenant) -> None:
    tenant_a, _ = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)
    session = factory()
    try:
        svc = __import__(
            "evenor.application.tenant_service", fromlist=["TenantService"]
        ).TenantService(
            __import__(
                "evenor.adapters.persistence.tenant_repository",
                fromlist=["SqlAlchemyTenantRepository"],
            ).SqlAlchemyTenantRepository(session)
        )
        b = svc.create_tenant(name="Other", slug="other-bot")
        session.commit()
        hooks_a = SqlAlchemyHookEventRepository(session, tenant_a.id)
        hooks_b = SqlAlchemyHookEventRepository(session, b.tenant.id)
        hooks_a.create(session_id="s", hook_type="test", payload_json="{}")
        session.commit()
        only_a = hooks_a.list_by_tenant()
        only_b = hooks_b.list_by_tenant()
    finally:
        session.close()
    assert len(only_a) == 1
    assert len(only_b) == 0
