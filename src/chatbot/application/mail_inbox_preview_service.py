from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from chatbot.adapters.mail.imap_client import ImapError, ImapMailClient, InboxPreviewMessage
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.mail_imap_uid_repository import SqlAlchemyMailImapUidRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.application.mail_connection_service import MailConnectionService
from chatbot.config.settings import Settings, get_settings
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorType
from chatbot.domain.models.mail_connection import MailConnection
from chatbot.mail.process_since import format_process_since_display, parse_process_since


@dataclass(frozen=True, slots=True)
class InboxPreviewItem:
    uid: str
    from_addr: str
    to_addr: str
    subject: str
    received_at: str | None
    body_preview: str
    eligible: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InboxPreviewResult:
    ok: bool
    message: str
    mailbox: str = ""
    process_since: str | None = None
    process_since_display: str = "—"
    messages: tuple[InboxPreviewItem, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "mailbox": self.mailbox,
            "process_since": self.process_since,
            "process_since_display": self.process_since_display,
            "messages": [m.to_dict() for m in self.messages],
            "error": self.error,
        }


def _skip_reason_for_message(
    *,
    mail: InboxPreviewMessage,
    process_since: datetime | None,
    uid_repo: SqlAlchemyMailImapUidRepository,
    draft_repo: SqlAlchemyMailDraftRepository,
) -> str | None:
    if uid_repo.exists_by_uid(mail.uid) or draft_repo.exists_by_uid(mail.uid):
        return "already recorded in database"
    if process_since is not None and mail.received_at is not None and mail.received_at < process_since:
        return f"before process_since ({process_since.astimezone(UTC).strftime('%Y-%m-%d %H:%M')} UTC)"
    if mail.body_preview == "(no text/plain or text/html body)":
        return "no text body (worker ignores this message)"
    return None


def _annotate_messages(
    session: Session,
    *,
    tenant_id: int,
    mails: list[InboxPreviewMessage],
    process_since: datetime | None,
) -> list[InboxPreviewItem]:
    uid_repo = SqlAlchemyMailImapUidRepository(session, tenant_id=tenant_id)
    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
    items: list[InboxPreviewItem] = []
    for mail in mails:
        skip_reason = _skip_reason_for_message(
            mail=mail,
            process_since=process_since,
            uid_repo=uid_repo,
            draft_repo=draft_repo,
        )
        received_at = (
            mail.received_at.astimezone(UTC).isoformat() if mail.received_at is not None else None
        )
        items.append(
            InboxPreviewItem(
                uid=mail.uid,
                from_addr=mail.from_addr,
                to_addr=mail.to_addr,
                subject=mail.subject,
                received_at=received_at,
                body_preview=mail.body_preview,
                eligible=skip_reason is None,
                skip_reason=skip_reason,
            )
        )
    return items


def _in_email_connector_for_mail_connection(
    session: Session,
    tenant_id: int,
    connection_id: int,
) -> Connector | None:
    for connector in ConnectorService(SqlAlchemyConnectorRepository(session)).list_for_tenant(
        tenant_id
    ):
        if (
            connector.type == ConnectorType.EMAIL
            and connector.direction == ConnectorDirection.IN
            and str(connector.config.get("mail_connection_id", "")).strip() == str(connection_id)
        ):
            return connector
    return None


def preview_mail_connection_inbox(
    connection: MailConnection,
    *,
    session: Session,
    tenant_id: int,
    settings: Settings | None = None,
    limit: int = 5,
) -> InboxPreviewResult:
    resolved_settings = settings or get_settings()
    in_connector = _in_email_connector_for_mail_connection(session, tenant_id, connection.id)
    process_since_raw = (
        str(in_connector.config.get("process_since", "")).strip() if in_connector else ""
    )
    process_since = parse_process_since({"process_since": process_since_raw}) if process_since_raw else None
    try:
        mail_cfg, _updated = MailConnectionService(session).resolve_runtime_config(
            connection, direction="in", settings=resolved_settings
        )
        client = ImapMailClient(mail_cfg)
        try:
            client.connect()
            mails = client.list_recent_messages(limit=limit)
        finally:
            client.close()
    except ImapError as exc:
        return InboxPreviewResult(ok=False, message="IMAP preview failed.", error=str(exc))

    mailbox = str(mail_cfg.get("username", "")).strip() or connection.email
    items = _annotate_messages(
        session, tenant_id=tenant_id, mails=mails, process_since=process_since
    )
    return InboxPreviewResult(
        ok=True,
        message=f"Latest {len(items)} message(s) in INBOX.",
        mailbox=mailbox,
        process_since=process_since_raw or None,
        process_since_display=format_process_since_display(process_since_raw or None),
        messages=tuple(items),
    )


def preview_tenant_inbox(
    session: Session,
    *,
    tenant_id: int,
    settings: Settings | None = None,
    limit: int = 5,
) -> InboxPreviewResult:
    connector = ConnectorService(SqlAlchemyConnectorRepository(session)).find(
        tenant_id, direction=ConnectorDirection.IN, type=ConnectorType.EMAIL
    )
    if connector is None or not connector.active:
        return InboxPreviewResult(
            ok=False,
            message="No active email inbound connector.",
            error="missing_in_connector",
        )
    raw_id = str(connector.config.get("mail_connection_id", "")).strip()
    if raw_id:
        connection = MailConnectionService(session).get_for_tenant(int(raw_id), tenant_id)
        if connection is not None:
            return preview_mail_connection_inbox(
                connection,
                session=session,
                tenant_id=tenant_id,
                settings=settings,
                limit=limit,
            )
    from chatbot.application.connector_test_service import _mail_config_for_test

    resolved_settings = settings or get_settings()
    process_since = parse_process_since(connector.config)
    try:
        mail_cfg = _mail_config_for_test(
            connector.config,
            direction="in",
            session=session,
            tenant_id=tenant_id,
            settings=resolved_settings,
        )
        client = ImapMailClient(mail_cfg)
        try:
            client.connect()
            mails = client.list_recent_messages(limit=limit)
        finally:
            client.close()
    except ImapError as exc:
        return InboxPreviewResult(ok=False, message="IMAP preview failed.", error=str(exc))

    mailbox = str(mail_cfg.get("username", "")).strip()
    items = _annotate_messages(
        session, tenant_id=tenant_id, mails=mails, process_since=process_since
    )
    return InboxPreviewResult(
        ok=True,
        message=f"Latest {len(items)} message(s) in INBOX.",
        mailbox=mailbox,
        process_since=str(connector.config.get("process_since", "")).strip() or None,
        process_since_display=format_process_since_display(
            str(connector.config.get("process_since", "")).strip() or None
        ),
        messages=tuple(items),
    )
