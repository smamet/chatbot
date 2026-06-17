from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session, sessionmaker

from chatbot.adapters.mail.imap_client import IncomingMail, imap_client
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.mail_imap_uid_repository import SqlAlchemyMailImapUidRepository
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.channel_outbound import get_outbound_connector
from chatbot.application.chat_service_factory import build_chat_service_for_worker
from chatbot.application.connector_service import ConnectorService
from chatbot.application.mail_connector_runtime import prepare_email_connector_config
from chatbot.application.outbound_orchestrator import queue_after_chat
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorType
from chatbot.domain.models.mail_draft import MailDraftStatus
from chatbot.mail.process_since import imap_since_date, parse_process_since, process_since_now_iso

logger = logging.getLogger(__name__)

_IMAP_TIMEOUT = 30


def _ensure_process_since(session: Session, in_connector: Connector) -> Connector:
    if parse_process_since(in_connector.config) is not None:
        return in_connector
    cfg = dict(in_connector.config)
    cfg["process_since"] = process_since_now_iso()
    repo = SqlAlchemyConnectorRepository(session)
    updated = repo.update(in_connector.id, config=cfg)
    session.flush()
    return updated or in_connector


def _uid_known(
    uid_repo: SqlAlchemyMailImapUidRepository,
    draft_repo: SqlAlchemyMailDraftRepository,
    uid: str,
) -> bool:
    return uid_repo.exists_by_uid(uid) or draft_repo.exists_by_uid(uid)


def _should_skip_mail(mail: IncomingMail, process_since: datetime | None) -> bool:
    if process_since is None or mail.received_at is None:
        return False
    return mail.received_at < process_since


def _process_one_mail(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    in_connector: Connector,
    mail: IncomingMail,
    imap_conn,
    uid_repo: SqlAlchemyMailImapUidRepository,
) -> bool:
    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
    if _uid_known(uid_repo, draft_repo, mail.uid):
        return False

    tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
    if tenant is None or not tenant.active:
        return False

    process_since = parse_process_since(in_connector.config)
    if _should_skip_mail(mail, process_since):
        uid_repo.record_skipped(mail.uid, received_at=mail.received_at)
        return False

    draft = draft_repo.create(
        imap_uid=mail.uid,
        from_addr=mail.from_addr,
        to_addr=mail.to_addr,
        subject=mail.subject,
        body_in=mail.body_text,
        status=MailDraftStatus.PENDING,
    )

    chat = build_chat_service_for_worker(session, settings, tenant)
    session_id = f"email:{mail.from_addr}"
    result = chat.handle_user_message(session_id, mail.body_text)

    connectors = ConnectorService(SqlAlchemyConnectorRepository(session))
    out_conn = get_outbound_connector(connectors, tenant_id, ConnectorType.EMAIL)
    if out_conn is None:
        draft_repo.mark_failed(draft.id, error="No active email outbound connector")
        return False

    queue_after_chat(
        session,
        tenant_id=tenant_id,
        connector=out_conn,
        session_id=session_id,
        recipient_id=mail.from_addr,
        result=result,
        settings=settings,
        tenant_slug=tenant.slug,
    )
    draft_repo.mark_processed(draft.id, draft_reply=result.text)
    uid_repo.record_processed(mail.uid, received_at=mail.received_at)
    imap_conn.mark_seen(mail.uid)
    return True


def _process_tenant_inbox(
    session: Session,
    *,
    settings: Settings,
    in_connector: Connector,
    tenant_slug: str | None = None,
) -> int:
    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=in_connector.tenant_id)
    uid_repo = SqlAlchemyMailImapUidRepository(session, tenant_id=in_connector.tenant_id)
    in_connector = _ensure_process_since(session, in_connector)
    process_since = parse_process_since(in_connector.config)
    since_date = imap_since_date(process_since) if process_since else None

    def skip_uid(uid: str) -> bool:
        return _uid_known(uid_repo, draft_repo, uid)

    processed = 0
    mail_config = prepare_email_connector_config(
        in_connector,
        session=session,
        direction=ConnectorDirection.IN,
        settings=settings,
    )
    with imap_client(mail_config, timeout=_IMAP_TIMEOUT) as imap:
        for mail in imap.fetch_pending(skip_uid, since_date=since_date):
            try:
                if _process_one_mail(
                    session,
                    settings=settings,
                    tenant_id=in_connector.tenant_id,
                    in_connector=in_connector,
                    mail=mail,
                    imap_conn=imap,
                    uid_repo=uid_repo,
                ):
                    session.commit()
                    processed += 1
                else:
                    session.commit()
            except Exception:
                session.rollback()
                logger.exception(
                    "Mail processing failed tenant_id=%s slug=%s uid=%s",
                    in_connector.tenant_id,
                    tenant_slug or "?",
                    mail.uid,
                )
    return processed


def run_once_for_tenant(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    tenant_id: int,
) -> int:
    with session_factory() as session:
        repo = SqlAlchemyConnectorRepository(session)
        connector = repo.find_active(
            tenant_id,
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
        )
        if connector is None:
            return 0
        tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
        slug = tenant.slug if tenant else None
        try:
            processed = _process_tenant_inbox(
                session,
                settings=settings,
                in_connector=connector,
                tenant_slug=slug,
            )
            session.commit()
            return processed
        except Exception:
            session.rollback()
            logger.exception(
                "Mail poll failed tenant_id=%s slug=%s",
                tenant_id,
                slug or "?",
            )
            return 0


def run_once(session_factory: sessionmaker[Session], settings: Settings) -> int:
    with session_factory() as session:
        connectors = SqlAlchemyConnectorRepository(session).list_active_by_type(
            direction=ConnectorDirection.IN,
            type=ConnectorType.EMAIL,
        )
    total = 0
    tenant_repo_factory = session_factory
    for connector in connectors:
        tenant_slug: str | None = None
        with tenant_repo_factory() as session:
            tenant = SqlAlchemyTenantRepository(session).find_by_id(connector.tenant_id)
            if tenant is None or not tenant.active:
                continue
            tenant_slug = tenant.slug
        try:
            with session_factory() as session:
                processed = _process_tenant_inbox(
                    session,
                    settings=settings,
                    in_connector=connector,
                    tenant_slug=tenant_slug,
                )
                session.commit()
                total += processed
        except Exception:
            logger.exception(
                "Mail poll failed tenant_id=%s slug=%s connector_id=%s",
                connector.tenant_id,
                tenant_slug or "?",
                connector.id,
            )
    return total
