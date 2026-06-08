from __future__ import annotations

import logging

from sqlalchemy.orm import Session, sessionmaker

from chatbot.adapters.mail.imap_client import IncomingMail, imap_client
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.application.channel_outbound import get_outbound_connector
from chatbot.application.chat_service_factory import build_chat_service_for_worker
from chatbot.application.connector_service import ConnectorService
from chatbot.application.outbound_orchestrator import queue_after_chat
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorType
from chatbot.domain.models.mail_draft import MailDraftStatus

logger = logging.getLogger(__name__)

_IMAP_TIMEOUT = 30


def _process_one_mail(
    session: Session,
    *,
    settings: Settings,
    tenant_id: int,
    in_connector: Connector,
    mail: IncomingMail,
    imap_conn,
) -> bool:
    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
    if draft_repo.exists_by_uid(mail.uid):
        return False

    draft = draft_repo.create(
        imap_uid=mail.uid,
        from_addr=mail.from_addr,
        to_addr=mail.to_addr,
        subject=mail.subject,
        body_in=mail.body_text,
        status=MailDraftStatus.PENDING,
    )

    tenant = SqlAlchemyTenantRepository(session).find_by_id(tenant_id)
    if tenant is None or not tenant.active:
        draft_repo.mark_failed(draft.id, error="Tenant inactive or missing")
        return False

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
    )
    draft_repo.mark_processed(draft.id, draft_reply=result.text)
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
    processed = 0
    with imap_client(in_connector.config, timeout=_IMAP_TIMEOUT) as imap:
        for mail in imap.fetch_pending(draft_repo.exists_by_uid):
            try:
                if _process_one_mail(
                    session,
                    settings=settings,
                    tenant_id=in_connector.tenant_id,
                    in_connector=in_connector,
                    mail=mail,
                    imap_conn=imap,
                ):
                    session.commit()
                    processed += 1
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
