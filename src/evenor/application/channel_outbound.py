from __future__ import annotations

from datetime import UTC, datetime

from evenor.adapters.channels import instagram_meta, messenger_meta, whatsapp_meta
from evenor.adapters.mail.body_format import email_draft_html_from_markdown
from evenor.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from evenor.adapters.persistence.email_thread_repository import SqlAlchemyEmailThreadRepository
from evenor.adapters.persistence.outbound_email_message_repository import (
    SqlAlchemyOutboundEmailMessageRepository,
)
from evenor.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from evenor.application.connector_service import ConnectorService
from evenor.application.email_outbound import (
    coalesce_stored_email_subject,
    resolve_email_subject,
    send_email_reply,
)
from evenor.application.email_threading import EmailThreadingContext, resolve_threading_for_reply
from evenor.application.mail_connector_runtime import prepare_email_connector_config
from evenor.application.mail_imap_seen_service import mark_imap_seen_for_pending_reply
from evenor.application.pending_reply_inbound import (
    inbound_for_pending_reply,
    inbound_subject_for_pending_reply,
)
from evenor.config.settings import Settings
from evenor.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from evenor.domain.models.fulfillment import FulfillmentKind
from evenor.domain.models.outbound_attachment import OutboundAttachment
from evenor.domain.models.pending_reply import PendingReply, PendingReplyStatus
from sqlalchemy.orm import Session


def get_outbound_connector(
    connectors: ConnectorService,
    tenant_id: int,
    channel: ConnectorType,
) -> Connector | None:
    out = connectors.find(tenant_id, direction=ConnectorDirection.OUT, type=channel)
    if out and out.active:
        return out
    conn = connectors.find(tenant_id, direction=ConnectorDirection.IN, type=channel)
    if conn and conn.active:
        return conn
    return None


def should_queue_for_validation(connector: Connector | None) -> bool:
    return connector is not None and connector.mode == ConnectorMode.VALIDATION


def queue_pending_reply(
    session: Session,
    *,
    tenant_id: int,
    connector_id: int,
    session_id: str,
    channel: str,
    recipient_id: str,
    draft_text: str,
    hook_event_id: int | None = None,
    fulfillment_kind: FulfillmentKind = FulfillmentKind.REPLY_ONLY,
    quote_proposal_json: str | None = None,
    quote_resolved_json: str | None = None,
    quote_external_id: str | None = None,
    attachments_json: str | None = None,
    draft_html: str | None = None,
    mail_draft_id: int | None = None,
    thread_id: int | None = None,
    inbound_email_subject: str | None = None,
) -> PendingReply:
    if draft_html is None and channel == ConnectorType.EMAIL.value:
        draft_html = email_draft_html_from_markdown(draft_text)
    draft_subject: str | None = None
    if channel == ConnectorType.EMAIL.value:
        inbound_subject = (inbound_email_subject or "").strip()
        if not inbound_subject:
            inbound_subject = inbound_subject_for_pending_reply(
                session,
                tenant_id,
                channel=channel,
                recipient_id=recipient_id,
                session_id=session_id,
                draft_text=draft_text,
                mail_draft_id=mail_draft_id,
            )
        conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        outbound = get_outbound_connector(conn_svc, tenant_id, ConnectorType.EMAIL)
        outbound_config = outbound.config if outbound else {}
        draft_subject = resolve_email_subject(
            connector_config=outbound_config,
            inbound_subject=inbound_subject or None,
        )
    return SqlAlchemyPendingReplyRepository(session).create(
        tenant_id=tenant_id,
        connector_id=connector_id,
        session_id=session_id,
        channel=channel,
        recipient_id=recipient_id,
        draft_text=draft_text,
        draft_html=draft_html,
        draft_subject=draft_subject,
        hook_event_id=hook_event_id,
        fulfillment_kind=fulfillment_kind,
        quote_proposal_json=quote_proposal_json,
        quote_resolved_json=quote_resolved_json,
        quote_external_id=quote_external_id,
        attachments_json=attachments_json,
        mail_draft_id=mail_draft_id,
        thread_id=thread_id,
    )


def dispatch_channel_reply(
    *,
    channel: str,
    recipient_id: str,
    text: str,
    config: dict,
    settings: Settings,
    attachments: list[OutboundAttachment] | None = None,
    body_html: str | None = None,
    subject: str | None = None,
    threading: EmailThreadingContext | None = None,
) -> str | None:
    if channel == ConnectorType.WHATSAPP.value:
        phone_id = str(config.get("phone_number_id", "")).strip()
        token = str(config.get("access_token", "")).strip() or settings.whatsapp_access_token
        if phone_id and token:
            if attachments:
                for att in attachments:
                    media_id = whatsapp_meta.upload_media(
                        phone_number_id=phone_id,
                        access_token=token,
                        data=att.data,
                        mime_type=att.mime_type,
                    )
                    whatsapp_meta.send_whatsapp_document(
                        phone_number_id=phone_id,
                        access_token=token,
                        to_wa_id=recipient_id,
                        media_id=media_id,
                        filename=att.filename,
                        caption=text if att == attachments[0] else None,
                    )
            else:
                whatsapp_meta.send_whatsapp_text(
                    phone_number_id=phone_id,
                    access_token=token,
                    to_wa_id=recipient_id,
                    text=text,
                )
        return None
    if channel == ConnectorType.MESSENGER.value:
        token = str(config.get("page_access_token", "")).strip() or settings.messenger_page_access_token
        if token:
            body = text
            if attachments:
                names = ", ".join(a.filename for a in attachments)
                body = f"{text}\n\n[Attachment: {names}]"
            messenger_meta.send_messenger_text(
                page_access_token=token,
                recipient_psid=recipient_id,
                text=body,
            )
        return None
    if channel == ConnectorType.INSTAGRAM.value:
        token = str(config.get("access_token", "")).strip() or settings.instagram_access_token
        ig_user = str(config.get("ig_user_id", "")).strip() or settings.instagram_ig_user_id
        if token and ig_user:
            body = text
            if attachments:
                names = ", ".join(a.filename for a in attachments)
                body = f"{text}\n\n[Attachment: {names}]"
            instagram_meta.send_instagram_text(
                access_token=token,
                ig_user_id=ig_user,
                recipient_igsid=recipient_id,
                text=body,
            )
        return None
    if channel == ConnectorType.EMAIL.value:
        return send_email_reply(
            config=config,
            to_addr=recipient_id,
            body=text,
            subject=subject,
            body_html=body_html,
            attachments=attachments,
            threading=threading,
        )
    return None


def approve_pending_reply(
    session: Session,
    reply: PendingReply,
    *,
    config: dict,
    settings: Settings,
    attachments: list[OutboundAttachment] | None = None,
) -> PendingReply | None:
    outbound_config = config
    threading = None
    send_subject = reply.draft_subject
    if reply.channel == ConnectorType.EMAIL.value:
        connector = SqlAlchemyConnectorRepository(session).find_by_id(reply.connector_id)
        if connector is not None:
            outbound_config = prepare_email_connector_config(
                connector,
                session=session,
                direction=ConnectorDirection.OUT,
                settings=settings,
                force_oauth_refresh=True,
            )
        inbound = inbound_for_pending_reply(session, reply.tenant_id, reply)
        send_subject = resolve_email_subject(
            draft_subject=coalesce_stored_email_subject(
                stored_draft_subject=reply.draft_subject,
                connector_config=outbound_config,
                inbound_subject=inbound.get("subject"),
            ),
            connector_config=outbound_config,
            inbound_subject=inbound.get("subject") or None,
        )
        threading = resolve_threading_for_reply(
            session,
            tenant_id=reply.tenant_id,
            reply=reply,
        )
    message_id = dispatch_channel_reply(
        channel=reply.channel,
        recipient_id=reply.recipient_id,
        text=reply.draft_text,
        config=outbound_config,
        settings=settings,
        attachments=attachments,
        body_html=reply.draft_html,
        subject=send_subject,
        threading=threading,
    )
    if reply.channel == ConnectorType.EMAIL.value and message_id and reply.thread_id is not None:
        SqlAlchemyOutboundEmailMessageRepository(session, tenant_id=reply.tenant_id).record(
            thread_id=reply.thread_id,
            message_id=message_id,
            in_reply_to=threading.in_reply_to if threading else None,
            references_header=threading.references if threading else None,
            pending_reply_id=reply.id,
            sent_at=datetime.now(UTC),
        )
        SqlAlchemyEmailThreadRepository(session, tenant_id=reply.tenant_id).touch_activity(
            reply.thread_id,
            datetime.now(UTC),
        )
    mark_imap_seen_for_pending_reply(
        session,
        tenant_id=reply.tenant_id,
        reply=reply,
        settings=settings,
    )
    return SqlAlchemyPendingReplyRepository(session).update_status(
        reply.id, PendingReplyStatus.APPROVED
    )


def persist_validation_email_subject(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    form_subject: str,
    outbound_config: dict,
) -> PendingReply:
    if reply.channel != ConnectorType.EMAIL.value:
        return reply
    inbound = inbound_for_pending_reply(session, tenant_id, reply)
    raw = str(form_subject).strip()
    subject = resolve_email_subject(
        draft_subject=coalesce_stored_email_subject(
            stored_draft_subject=raw or None,
            connector_config=outbound_config,
            inbound_subject=inbound.get("subject"),
        ),
        connector_config=outbound_config,
        inbound_subject=inbound.get("subject") or None,
    )
    if subject == (reply.draft_subject or ""):
        return reply
    updated = SqlAlchemyPendingReplyRepository(session).update_draft(
        reply.id, draft_subject=subject
    )
    return updated or reply
