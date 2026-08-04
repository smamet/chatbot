from __future__ import annotations

import logging
from datetime import UTC

from sqlalchemy.orm import Session

from evenor.adapters.mail.imap_client import ImapError, imap_client
from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.application.mail_connector_runtime import prepare_email_connector_config
from evenor.config.settings import Settings
from evenor.domain.models.connector import ConnectorDirection, ConnectorType
from evenor.domain.models.pending_reply import PendingReply

logger = logging.getLogger(__name__)

_IMAP_TIMEOUT = 30


def mark_imap_seen_for_pending_reply(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    settings: Settings,
) -> None:
    if (reply.channel or "").lower() != ConnectorType.EMAIL.value:
        return
    draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_for_pending_reply(reply)
    if draft is None or not draft.imap_uid.strip():
        logger.warning("No IMAP UID for pending_reply id=%s tenant_id=%s", reply.id, tenant_id)
        return
    in_connector = SqlAlchemyConnectorRepository(session).find_active(
        tenant_id,
        direction=ConnectorDirection.IN,
        type=ConnectorType.EMAIL,
    )
    if in_connector is None:
        logger.warning(
            "No active inbound email connector for tenant_id=%s (pending_reply id=%s)",
            tenant_id,
            reply.id,
        )
        return
    mail_config = prepare_email_connector_config(
        in_connector,
        session=session,
        direction=ConnectorDirection.IN,
        settings=settings,
    )
    try:
        with imap_client(mail_config, timeout=_IMAP_TIMEOUT) as imap:
            imap.mark_seen(draft.imap_uid)
    except ImapError:
        logger.exception(
            "Failed to mark IMAP UID %s seen for pending_reply id=%s",
            draft.imap_uid,
            reply.id,
        )