from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, delete, func, select
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

BACKUP_VERSION = 1
MANIFEST_NAME = "manifest.json"
OPERATIONAL_NAME = "operational.json"

_OPERATIONAL_TABLES: tuple[tuple[str, type], ...] = (
    ("messages", MessageRow),
    ("hook_events", HookEventRow),
    ("orders", OrderRow),
    ("order_events", OrderEventRow),
    ("pending_replies", PendingReplyRow),
    ("pending_reply_edits", PendingReplyEditRow),
    ("pending_reply_audit_events", PendingReplyAuditEventRow),
    ("mail_drafts", MailDraftRow),
    ("test_chat_sessions", TestChatSessionRow),
)


class TenantFlushError(RuntimeError):
    pass


def default_backup_dir(settings: Settings, slug: str, *, at: datetime | None = None) -> Path:
    ts = (at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return settings.data_root / "backups" / slug / ts


def _serialize_row(row: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in row.__table__.columns:
        val = getattr(row, col.name)
        if isinstance(val, datetime):
            out[col.name] = val.isoformat()
        else:
            out[col.name] = val
    return out


def _deserialize_row(model: type, data: dict[str, Any]) -> Any:
    parsed: dict[str, Any] = {}
    for col in model.__table__.columns:
        if col.name not in data:
            continue
        val = data[col.name]
        if val is None:
            parsed[col.name] = None
        elif isinstance(col.type, DateTime) and isinstance(val, str):
            parsed[col.name] = datetime.fromisoformat(val.replace("Z", "+00:00"))
        else:
            parsed[col.name] = val
    return model(**parsed)


class TenantFlushService:
    def __init__(self, session: Session, *, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._tenant_repo = SqlAlchemyTenantRepository(session)

    def flush(self, slug: str, *, backup: bool = True) -> tuple[list[str], Path | None]:
        tenant = self._tenant_repo.find_by_slug(slug)
        if tenant is None:
            raise TenantFlushError(f"Unknown tenant slug: {slug}")

        tenant_id = tenant.id
        logs: list[str] = []
        backup_path: Path | None = None

        if backup:
            backup_path = self.create_backup(slug)
            logs.append(f"backup saved: {backup_path}")

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
        return logs, backup_path

    def create_backup(self, slug: str) -> Path:
        tenant = self._tenant_repo.find_by_slug(slug)
        if tenant is None:
            raise TenantFlushError(f"Unknown tenant slug: {slug}")
        return self._write_backup(slug, tenant.id)

    def restore(self, slug: str, backup_path: Path) -> list[str]:
        tenant = self._tenant_repo.find_by_slug(slug)
        if tenant is None:
            raise TenantFlushError(f"Unknown tenant slug: {slug}")
        if not backup_path.is_dir():
            raise TenantFlushError(f"Backup directory not found: {backup_path}")

        manifest_path = backup_path / MANIFEST_NAME
        operational_path = backup_path / OPERATIONAL_NAME
        if not manifest_path.is_file() or not operational_path.is_file():
            raise TenantFlushError("Invalid backup: missing manifest.json or operational.json")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("backup_version") != BACKUP_VERSION:
            raise TenantFlushError(
                f"Unsupported backup version: {manifest.get('backup_version')}"
            )
        if manifest.get("slug") != slug:
            raise TenantFlushError(
                f"Backup slug {manifest.get('slug')!r} does not match {slug!r}"
            )
        if manifest.get("tenant_id") != tenant.id:
            raise TenantFlushError(
                "Backup tenant_id does not match current bot — restore only to the same bot"
            )

        payload = json.loads(operational_path.read_text(encoding="utf-8"))
        logs: list[str] = []

        logs.extend(self.flush(slug, backup=False)[0])
        logs.extend(self._insert_operational_rows(payload))
        logs.extend(self._restore_runtime_dirs(slug, backup_path))

        self._session.flush()
        return logs

    def _write_backup(self, slug: str, tenant_id: int) -> Path:
        backup_path = default_backup_dir(self._settings, slug)
        backup_path.mkdir(parents=True, exist_ok=False)

        payload: dict[str, list[dict[str, Any]]] = {}
        for key, model in _OPERATIONAL_TABLES:
            rows = list(
                self._session.scalars(select(model).where(model.tenant_id == tenant_id))
            )
            payload[key] = [_serialize_row(row) for row in rows]
        order_ids = list(
            self._session.scalars(select(OrderRow.id).where(OrderRow.tenant_id == tenant_id))
        )
        if order_ids:
            payload["order_items"] = [
                _serialize_row(row)
                for row in self._session.scalars(
                    select(OrderItemRow).where(OrderItemRow.order_id.in_(order_ids))
                )
            ]
        else:
            payload["order_items"] = []

        manifest = {
            "backup_version": BACKUP_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "slug": slug,
            "tenant_id": tenant_id,
        }
        (backup_path / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (backup_path / OPERATIONAL_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        for name in ("attachments", "quotes"):
            src = self._settings.data_root / name / slug
            if src.is_dir():
                shutil.copytree(src, backup_path / name)
        return backup_path

    def _insert_operational_rows(self, payload: dict[str, list[dict[str, Any]]]) -> list[str]:
        logs: list[str] = []
        insert_order: list[tuple[str, type]] = [
            ("messages", MessageRow),
            ("hook_events", HookEventRow),
            ("orders", OrderRow),
            ("order_items", OrderItemRow),
            ("order_events", OrderEventRow),
            ("pending_replies", PendingReplyRow),
            ("pending_reply_edits", PendingReplyEditRow),
            ("pending_reply_audit_events", PendingReplyAuditEventRow),
            ("mail_drafts", MailDraftRow),
            ("test_chat_sessions", TestChatSessionRow),
        ]
        for key, model in insert_order:
            rows = payload.get(key, [])
            for row_data in rows:
                self._session.add(_deserialize_row(model, row_data))
            logs.append(f"restored {key}: {len(rows)}")
        return logs

    def _restore_runtime_dirs(self, slug: str, backup_path: Path) -> list[str]:
        logs: list[str] = []
        for name in ("attachments", "quotes"):
            src = backup_path / name
            if not src.is_dir():
                continue
            dest = self._settings.data_root / name / slug
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            logs.append(f"restored {name}: {dest}")
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
