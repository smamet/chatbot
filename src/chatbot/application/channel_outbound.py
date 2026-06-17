from __future__ import annotations

from chatbot.adapters.channels import instagram_meta, messenger_meta, whatsapp_meta
from chatbot.adapters.mail.body_format import email_draft_html_from_markdown
from chatbot.application.email_outbound import send_email_reply
from chatbot.application.mail_connector_runtime import prepare_email_connector_config
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.connector_service import ConnectorService
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.outbound_attachment import OutboundAttachment
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus
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
) -> PendingReply:
    if draft_html is None and channel == ConnectorType.EMAIL.value:
        draft_html = email_draft_html_from_markdown(draft_text)
    return SqlAlchemyPendingReplyRepository(session).create(
        tenant_id=tenant_id,
        connector_id=connector_id,
        session_id=session_id,
        channel=channel,
        recipient_id=recipient_id,
        draft_text=draft_text,
        draft_html=draft_html,
        hook_event_id=hook_event_id,
        fulfillment_kind=fulfillment_kind,
        quote_proposal_json=quote_proposal_json,
        quote_resolved_json=quote_resolved_json,
        quote_external_id=quote_external_id,
        attachments_json=attachments_json,
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
) -> None:
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
        return
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
        return
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
        return
    if channel == ConnectorType.EMAIL.value:
        send_email_reply(
            config=config,
            to_addr=recipient_id,
            body=text,
            body_html=body_html,
            attachments=attachments,
        )


def approve_pending_reply(
    session: Session,
    reply: PendingReply,
    *,
    config: dict,
    settings: Settings,
    attachments: list[OutboundAttachment] | None = None,
) -> PendingReply | None:
    outbound_config = config
    if reply.channel == ConnectorType.EMAIL.value:
        connector = SqlAlchemyConnectorRepository(session).find_by_id(reply.connector_id)
        if connector is not None:
            outbound_config = prepare_email_connector_config(
                connector,
                session=session,
                direction=ConnectorDirection.OUT,
                settings=settings,
            )
    dispatch_channel_reply(
        channel=reply.channel,
        recipient_id=reply.recipient_id,
        text=reply.draft_text,
        config=outbound_config,
        settings=settings,
        attachments=attachments,
        body_html=reply.draft_html,
    )
    return SqlAlchemyPendingReplyRepository(session).update_status(
        reply.id, PendingReplyStatus.APPROVED
    )
