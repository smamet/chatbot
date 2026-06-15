from __future__ import annotations

import shutil

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import (
    ConnectorRow,
    HookEventRow,
    IngestedFileRow,
    IntegrationRow,
    MailDraftRow,
    MailImapUidRow,
    MessageRow,
    OrderEventRow,
    OrderItemRow,
    OrderRow,
    PendingReplyAuditEventRow,
    PendingReplyEditRow,
    PendingReplyRow,
    TenantRow,
    TestChatSessionRow,
)
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.config.settings import Settings


class TenantFlushError(RuntimeError):
    pass


class TenantFlushService:
    def __init__(self, session: Session, *, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._tenant_repo = SqlAlchemyTenantRepository(session)

    def flush(self, slug: str) -> list[str]:
        tenant = self._tenant_repo.find_by_slug(slug)
        if tenant is None:
            raise TenantFlushError(f"Unknown tenant slug: {slug}")

        tenant_id = tenant.id
        logs: list[str] = []

        logs.append(self._delete_pending_reply_edits(tenant_id))
        logs.append(self._delete_pending_reply_audit(tenant_id))
        logs.append(self._delete_pending_replies(tenant_id))
        logs.append(self._delete_hook_events(tenant_id))
        logs.extend(self._delete_orders(tenant_id))
        logs.append(self._delete_messages(tenant_id))
        logs.append(self._delete_mail_drafts(tenant_id))
        logs.append(self._delete_test_chat_sessions(tenant_id))
        logs.extend(self._remove_runtime_dirs(slug))
        logs.extend(self._summarize_kept(tenant_id))

        self._session.flush()
        return logs

    def _count_delete(self, stmt) -> str:
        result = self._session.execute(stmt)
        return str(result.rowcount or 0)

    def _delete_pending_reply_edits(self, tenant_id: int) -> str:
        n = self._count_delete(
            delete(PendingReplyEditRow).where(PendingReplyEditRow.tenant_id == tenant_id)
        )
        return f"deleted pending_reply_edits: {n}"

    def _delete_pending_reply_audit(self, tenant_id: int) -> str:
        n = self._count_delete(
            delete(PendingReplyAuditEventRow).where(
                PendingReplyAuditEventRow.tenant_id == tenant_id
            )
        )
        return f"deleted pending_reply_audit_events: {n}"

    def _delete_pending_replies(self, tenant_id: int) -> str:
        n = self._count_delete(
            delete(PendingReplyRow).where(PendingReplyRow.tenant_id == tenant_id)
        )
        return f"deleted pending_replies: {n}"

    def _delete_hook_events(self, tenant_id: int) -> str:
        n = self._count_delete(delete(HookEventRow).where(HookEventRow.tenant_id == tenant_id))
        return f"deleted hook_events: {n}"

    def _delete_orders(self, tenant_id: int) -> list[str]:
        order_ids = list(
            self._session.scalars(select(OrderRow.id).where(OrderRow.tenant_id == tenant_id))
        )
        if not order_ids:
            return ["deleted orders: 0"]
        item_n = self._count_delete(
            delete(OrderItemRow).where(OrderItemRow.order_id.in_(order_ids))
        )
        event_n = self._count_delete(
            delete(OrderEventRow).where(OrderEventRow.tenant_id == tenant_id)
        )
        order_n = self._count_delete(delete(OrderRow).where(OrderRow.tenant_id == tenant_id))
        return [
            f"deleted order_items: {item_n}",
            f"deleted order_events: {event_n}",
            f"deleted orders: {order_n}",
        ]

    def _delete_messages(self, tenant_id: int) -> str:
        n = self._count_delete(delete(MessageRow).where(MessageRow.tenant_id == tenant_id))
        return f"deleted messages: {n}"

    def _delete_mail_drafts(self, tenant_id: int) -> str:
        n = self._count_delete(delete(MailDraftRow).where(MailDraftRow.tenant_id == tenant_id))
        return f"deleted mail_drafts: {n}"

    def _delete_test_chat_sessions(self, tenant_id: int) -> str:
        n = self._count_delete(
            delete(TestChatSessionRow).where(TestChatSessionRow.tenant_id == tenant_id)
        )
        return f"deleted test_chat_sessions: {n}"

    def _remove_runtime_dirs(self, slug: str) -> list[str]:
        logs: list[str] = []
        for name in ("attachments", "quotes"):
            root = self._settings.data_root / name / slug
            if root.exists():
                shutil.rmtree(root)
                logs.append(f"removed {root}")
        return logs

    def _summarize_kept(self, tenant_id: int) -> list[str]:
        ingested = self._session.scalar(
            select(func.count()).select_from(IngestedFileRow).where(
                IngestedFileRow.tenant_id == tenant_id
            )
        )
        connectors = self._session.scalar(
            select(func.count()).select_from(ConnectorRow).where(ConnectorRow.tenant_id == tenant_id)
        )
        integrations = self._session.scalar(
            select(func.count()).select_from(IntegrationRow).where(
                IntegrationRow.tenant_id == tenant_id
            )
        )
        imap_uids = self._session.scalar(
            select(func.count()).select_from(MailImapUidRow).where(
                MailImapUidRow.tenant_id == tenant_id
            )
        )
        tenant_ok = self._session.get(TenantRow, tenant_id) is not None
        return [
            f"kept tenant: {tenant_ok}",
            f"kept ingested_files: {ingested or 0}",
            f"kept connectors: {connectors or 0}",
            f"kept integrations: {integrations or 0}",
            f"kept mail_imap_uids: {imap_uids or 0}",
        ]
