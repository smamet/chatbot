from __future__ import annotations

import asyncio
import json
import logging
import uuid
import threading
from urllib.parse import quote as url_quote
from datetime import UTC
from decimal import Decimal, InvalidOperation
from pathlib import Path
from queue import Empty, Queue

import httpx

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.api_usage_repository import SqlAlchemyApiUsageRepository, usage_since_date
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir
from chatbot.adapters.persistence.test_chat_session_repository import TestChatSessionRepository
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.bot_bundle_service import (
    BotBundleError,
    ImportMode,
    build_export,
    import_bundle,
)
from chatbot.adapters.erpnext.client import ErpNextClient
from chatbot.adapters.mail.body_format import email_draft_html_from_markdown
from chatbot.application.channel_outbound import (
    approve_pending_reply,
    get_outbound_connector,
    persist_validation_email_subject,
)
from chatbot.application.email_outbound import coalesce_stored_email_subject, resolve_email_subject
from chatbot.application.email_thread_resolution import audit_from_json, format_resolution_tooltip
from chatbot.application.pending_reply_inbound import inbound_for_pending_reply
from chatbot.application.validation_message_ui import (
    ValidationMessageBubble,
    bubble_for_session_message,
    bubble_from_content,
    bubble_from_draft,
    bubble_to_ui_dict,
    email_from_addr_from_session,
    messages_to_bubbles,
)
from chatbot.application.validation_regenerate_service import (
    ValidationRegenerateError,
    generate_pending_reply_from_raw,
)
from chatbot.application.validation_translate_service import (
    ValidationTranslateError,
    translate_pending_reply_draft,
)
from chatbot.application.customer_access_gate import build_channel_session_id, resolve_manual_identity
from chatbot.application.customer_provisioning_service import create_erpnext_customer_for_test
from chatbot.application.erpnext_catalog_sync_service import (
    apply_catalog_rag_transition,
    catalog_rag_effective_enabled,
    purge_catalog_files_and_rag,
    sync_erpnext_catalog_for_tenant,
    update_catalog_sync_metadata,
)
from chatbot.application.quote_test_service import create_erpnext_quotation_for_test
from chatbot.application.hook_prompt_composer import compose_hook_instructions
from chatbot.application.monitoring_dashboard_service import MonitoringDashboardService
from chatbot.application.monitoring_date_range import (
    monitoring_query_string,
    parse_monitoring_range,
)
from chatbot.application.monitoring_format import format_count, format_usd
from chatbot.application.disk_usage_service import DiskUsageService, format_bytes
from chatbot.application.draft_edit_service import DraftEditError, save_pending_reply_draft
from chatbot.application.outbound_orchestrator import erpnext_integration_for_tenant, queue_after_chat
from chatbot.application.progress_log import ProgressLog
from chatbot.application.product_resolver import resolved_lines_from_json, resolved_lines_to_json
from chatbot.application.quote_fulfillment_service import (
    QuoteFulfillmentError,
    QuoteFulfillmentService,
    all_lines_resolved,
    create_quote_for_session,
    refresh_quote_pdf,
    resolve_quote_hook,
)
from chatbot.application.quote_pdf_storage import (
    AttachmentValidationError,
    attachment_rows_for_ui,
    cleanup_pending_reply_attachments,
    is_quote_pdf_path,
    is_user_attachment_path,
    merge_attachment_entries,
    partition_attachment_entries,
    quote_pdf_dashboard_url,
    quote_pdf_path,
    remove_attachment_entry,
    safe_quote_filename,
    store_outbound_attachment,
    validate_outbound_attachment_upload,
)
from chatbot.application.quote_sync_state import quote_pdf_stale_context
from chatbot.automation.modules.registry import all_modules
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.domain.models.message import ChatMessage
from chatbot.application.connector_service import ConnectorService
from chatbot.adapters.google.oauth import (
    build_authorize_url as google_build_authorize_url,
    exchange_code as google_exchange_code,
)
from chatbot.adapters.microsoft.oauth import (
    build_authorize_url as microsoft_build_authorize_url,
    exchange_code as microsoft_exchange_code,
)
from chatbot.adapters.oauth_state import (
    sign_connector_oauth_state,
    sign_mail_connection_oauth_state,
    verify_connector_oauth_state,
    verify_mail_connection_oauth_state,
)
from chatbot.application.connector_test_service import run_connector_connection_test, run_mail_connection_test
from chatbot.application.mail_inbox_preview_service import preview_mail_connection_inbox
from chatbot.application.mail_imap_seen_service import mark_imap_seen_for_pending_reply
from chatbot.application.mail_connection_service import (
    MailConnectionError,
    MailConnectionService,
    connector_auth_type_for_connection,
    connection_client_credentials,
    strip_connector_oauth_fields,
)
from chatbot.application.mail_oauth_service import (
    MailOAuthError,
    apply_oauth_tokens_to_config,
    is_oauth_connected,
    platform_google_mail_oauth_configured,
    platform_microsoft_mail_oauth_configured,
    resolve_mail_oauth_credentials,
)
from chatbot.adapters.quickbooks.oauth import (
    build_authorize_url,
    exchange_code,
    sign_oauth_state,
    verify_oauth_state,
)
from chatbot.application.email_test_service import (
    EmailTestError,
    get_email_test_connectors,
    inject_test_email,
    poll_tenant_now,
)
from chatbot.domain.models.mail_connection import MailConnectionProvider
from chatbot.application.integration_test_service import run_integration_test
from chatbot.application.integration_service import IntegrationService
from chatbot.application.sync_service import IngestSyncService
from chatbot.application.tenant_service import TenantService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.application.usage_metering import metered_embedder
from chatbot.application.usage_recorder_service import UsageRecorderService
from chatbot.application.user_service import UserService
from chatbot.application.validation_audit_service import VALIDATION_ACTIVITY_LIMIT, ValidationAuditService
from chatbot.config.settings import Settings
from chatbot.domain.constants import DEFAULT_HOOK_INSTRUCTIONS
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.connector_schema import (
    EmailAuthType,
    EmailOutboundProvider,
    connector_schemas_for_template,
    fields_for,
    oauth_managed_connector_keys,
    secret_connector_keys,
)
from chatbot.domain.models.hook import HookStatus
from chatbot.domain.models.integration import Integration, IntegrationType
from chatbot.domain.models.integration_schema import (
    fields_for as integration_fields_for,
    integration_meta_for_template,
    integration_schemas_for_template,
    is_quickbooks_connected,
    secret_integration_keys,
)
from chatbot.domain.models.pending_reply import PendingReply, PendingReplyStatus
from chatbot.domain.models.pending_reply_audit import ValidationAuditAction
from chatbot.domain.models.tenant import Tenant, TenantConfig
from chatbot.domain.models.user import User, UserRole
from chatbot.mail.process_since import (
    format_for_datetime_local,
    parse_from_form,
    process_since_now_iso,
)
from chatbot.interfaces.api.deps import (
    _build_chat_service,
    get_session,
    get_settings_dep,
    get_tenant_service,
)
from chatbot.interfaces.web.deps import (
    get_user_service,
    reject_validation_only,
    require_admin,
    require_editor,
    require_user,
    require_validator,
)
from chatbot.interfaces.web.templates import templates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_WEBHOOK_CHANNELS = frozenset(
    {ConnectorType.WHATSAPP, ConnectorType.MESSENGER, ConnectorType.INSTAGRAM}
)


def _status_class(status: str) -> str:
    return {
        HookStatus.PENDING.value: "pending",
        HookStatus.PROCESSING.value: "processing",
        HookStatus.DONE.value: "done",
        HookStatus.FAILED.value: "failed",
    }.get(status, "")


def _tenant_or_404(tenant_service: TenantService, slug: str) -> Tenant:
    tenant = tenant_service.get_by_slug(slug)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return tenant


def _require_access(user: User, user_service: UserService, tenant: Tenant) -> None:
    if not user_service.can_access_tenant(user, tenant.id):
        raise HTTPException(status_code=403, detail="Forbidden")


def _merge_integration_config(existing: dict | None, incoming: dict) -> dict:
    secrets = secret_integration_keys()
    base = dict(existing or {})
    for key, value in incoming.items():
        if key in secrets and not str(value).strip():
            continue
        base[key] = value
    return base


def _integration_config_from_form(integration_type: str, fields: dict[str, str]) -> dict:
    schema_fields = integration_fields_for(integration_type)
    secrets = secret_integration_keys()
    raw = {field.key: fields.get(field.key, "").strip() for field in schema_fields}
    cfg = {key: value for key, value in raw.items() if value or key not in secrets}
    for field in schema_fields:
        if field.input_type == "checkbox":
            cfg[field.key] = fields.get(field.key, "") == "on"
        elif field.input_type == "number" and field.key in cfg:
            try:
                cfg[field.key] = int(cfg[field.key])
            except ValueError:
                cfg[field.key] = int(field.default or "0")
    return cfg


def _public_base_url(request: Request, settings: Settings) -> str:
    public = settings.public_base_url.strip()
    if public:
        return public.rstrip("/")
    return str(request.base_url).rstrip("/")


def _quickbooks_redirect_uri(request: Request, settings: Settings, slug: str) -> str:
    return (
        f"{_public_base_url(request, settings)}"
        f"/dashboard/bots/{slug}/integrations/quickbooks/callback"
    )


def _mail_oauth_redirect_uri(request: Request, settings: Settings, slug: str, provider: str) -> str:
    return (
        f"{_public_base_url(request, settings)}"
        f"/dashboard/bots/{slug}/connectors/{provider}/callback"
    )


def _platform_mail_oauth_redirect_uri(request: Request, settings: Settings) -> str:
    return f"{_public_base_url(request, settings)}/dashboard/mail-oauth/callback"


def _slug_from_mail_oauth_state(state: str, settings: Settings) -> str:
    if not state.strip():
        return ""
    try:
        state_data = verify_mail_connection_oauth_state(state, secret=_oauth_signing_secret(settings))
    except ValueError:
        return ""
    return str(state_data.get("slug", "")).strip()


def _normalize_email_connector_config(cfg: dict, *, session: Session, tenant_id: int) -> dict:
    raw = cfg.get("mail_connection_id")
    if raw is None or str(raw).strip() == "":
        return cfg
    try:
        connection_id = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid mail connection") from exc
    connection = MailConnectionService(session).get_for_tenant(connection_id, tenant_id)
    if connection is None:
        raise HTTPException(status_code=400, detail="Mail connection not found")
    normalized = strip_connector_oauth_fields(dict(cfg))
    normalized["mail_connection_id"] = connection_id
    normalized["auth_type"] = connector_auth_type_for_connection(connection.provider)
    return normalized


def _oauth_signing_secret(settings: Settings) -> str:
    secret = settings.session_secret.strip() or settings.app_secret_key.strip()
    if not secret:
        raise HTTPException(status_code=503, detail="SESSION_SECRET is required for OAuth")
    return secret


async def _connector_config_from_request(
    form,
    *,
    tenant_id: int,
    session: Session,
    connector_type: str,
    direction: str,
) -> tuple[ConnectorType, ConnectorDirection, ConnectorMode, dict, str | None]:
    try:
        ctype = ConnectorType(connector_type)
        cdir = ConnectorDirection(direction)
        cmode = (
            ConnectorMode.DIRECT
            if cdir == ConnectorDirection.IN
            else ConnectorMode(str(form.get("mode", "validation")))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid connector") from exc
    outbound_provider: str | None = None
    if ctype == ConnectorType.EMAIL and cdir == ConnectorDirection.OUT:
        outbound_provider = (
            str(form.get("outbound_provider", EmailOutboundProvider.SMTP.value)).strip()
            or EmailOutboundProvider.SMTP.value
        )
    schema_fields = fields_for(connector_type, direction, outbound_provider=outbound_provider)
    field_values = {field.key: str(form.get(field.key, "")).strip() for field in schema_fields}
    for field in schema_fields:
        if field.input_type == "checkbox":
            field_values[field.key] = "on" if form.get(field.key) == "on" else ""
    incoming = _connector_config_from_form(
        connector_type, direction, field_values, outbound_provider=outbound_provider
    )
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    existing = svc.find(tenant_id, direction=cdir, type=ctype)
    cfg = _merge_connector_config(existing.config if existing else None, incoming)
    if ctype == ConnectorType.EMAIL and cdir == ConnectorDirection.IN:
        if not str(cfg.get("process_since", "")).strip():
            cfg["process_since"] = process_since_now_iso()
    if ctype == ConnectorType.EMAIL and cdir == ConnectorDirection.OUT:
        cfg["outbound_provider"] = outbound_provider or EmailOutboundProvider.SMTP.value
    if ctype == ConnectorType.EMAIL and not str(cfg.get("auth_type", "")).strip():
        cfg["auth_type"] = EmailAuthType.PASSWORD.value
    if ctype == ConnectorType.EMAIL:
        cfg = _normalize_email_connector_config(cfg, session=session, tenant_id=tenant_id)
    return ctype, cdir, cmode, cfg, outbound_provider


def _integration_endpoint(integration: Integration) -> str:
    if integration.type == IntegrationType.ERPNEXT:
        return str(integration.config.get("url") or "—")
    if integration.type == IntegrationType.QUICKBOOKS:
        realm = str(integration.config.get("realm_id") or "").strip()
        return f"realm {realm}" if realm else "Not connected"
    return "—"


def _active_integration_types(integrations: list[Integration]) -> set[str]:
    return {i.type.value for i in integrations if i.active}


def _automation_modules_for_ui(integrations: list[Integration]) -> list[dict]:
    active = _active_integration_types(integrations)
    rows: list[dict] = []
    for mod in all_modules():
        available = mod.requires_integration is None or mod.requires_integration in active
        rows.append(
            {
                "id": mod.id,
                "label": mod.label,
                "description": mod.description,
                "ui_enabled": mod.ui_enabled,
                "available": available,
                "requires_integration": mod.requires_integration,
            }
        )
    return rows


def _email_from_addr_for_reply(reply: PendingReply) -> str:
    addr = email_from_addr_from_session(reply.session_id)
    if addr:
        return addr
    return (reply.recipient_id or "").strip().lower()


def _message_bubble_for_reply(
    session: Session,
    tenant_id: int,
    reply: PendingReply,
    *,
    role,
    content: str,
    created_at,
) -> ValidationMessageBubble:
    from chatbot.domain.models.message import MessageRole

    if (reply.channel or "").lower() != "email" or role != MessageRole.USER:
        return bubble_from_content(role=role, content=content, created_at=created_at)

    draft_repo = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
    draft = draft_repo.find_nearest_before(
        thread_id=reply.thread_id,
        from_addr=_email_from_addr_for_reply(reply),
        before=created_at,
    )
    if draft is not None:
        return bubble_from_draft(draft=draft, role=role, created_at=created_at)
    return bubble_from_content(role=role, content=content, created_at=created_at)


def _conversation_history_for_pending_reply(
    session: Session,
    tenant_id: int,
    reply: PendingReply,
) -> list[ValidationMessageBubble]:
    from chatbot.domain.models.message import MessageRole

    conv = SqlAlchemyConversationRepository(session, tenant_id)
    before = reply.created_at
    if before.tzinfo is None:
        before = before.replace(tzinfo=UTC)
    messages = conv.list_messages_before(reply.session_id, before)
    if messages:
        messages = messages[:-1]
    if messages:
        return [
            _message_bubble_for_reply(
                session,
                tenant_id,
                reply,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ]
    inbound = inbound_for_pending_reply(session, tenant_id, reply)
    if inbound.get("text"):
        draft = inbound.get("mail_draft")
        if draft is not None:
            return [
                bubble_from_draft(
                    draft=draft,
                    role=MessageRole.USER,
                    created_at=inbound.get("received_at"),
                )
            ]
        return [
            bubble_from_content(
                role=MessageRole.USER,
                content=inbound["text"],
                created_at=inbound.get("received_at"),
                content_raw=inbound.get("raw_text") or None,
            )
        ]
    return []


def _inbound_for_pending_reply(session: Session, tenant_id: int, reply) -> dict:
    return inbound_for_pending_reply(session, tenant_id, reply)


def _outbound_email_config(session: Session, tenant_id: int) -> dict:
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    outbound = get_outbound_connector(conn_svc, tenant_id, ConnectorType.EMAIL)
    return outbound.config if outbound else {}


def _erpnext_quotation_edit_url(session: Session, tenant_id: int, quote_name: str | None) -> str | None:
    if not quote_name:
        return None
    integration = erpnext_integration_for_tenant(session, tenant_id)
    if integration is None:
        return None
    base = str(integration[1].get("url", "")).strip().rstrip("/")
    if not base:
        return None
    safe_name = url_quote(quote_name.strip(), safe="")
    return f"{base}/app/quotation/{safe_name}"


def _validation_detail_url(slug: str, reply_id: int) -> str:
    return f"/dashboard/bots/{slug}/validation/{reply_id}"


def _validation_inbox_url(slug: str, *, vsub: str = "pending") -> str:
    return f"/dashboard/bots/{slug}?tab=validation&vsub={vsub}"


def _quote_pdf_stale_info(
    session: Session,
    tenant_id: int,
    reply: PendingReply,
    *,
    tenant_slug: str,
) -> dict:
    quote_name = (reply.quote_external_id or "").strip()
    erpnext_url = _erpnext_quotation_edit_url(session, tenant_id, quote_name) if quote_name else None
    integration = erpnext_integration_for_tenant(session, tenant_id)
    client = integration[0] if integration else None
    return quote_pdf_stale_context(
        client=client,
        tenant_slug=tenant_slug,
        quote_name=quote_name or None,
        stored_modified=reply.quote_erp_modified,
        erpnext_url=erpnext_url,
    )


def _pending_reply_ui_item(
    session: Session,
    tenant_id: int,
    reply: PendingReply,
    *,
    tenant_slug: str,
    settings: Settings,
) -> dict:
    resolved = resolved_lines_from_json(reply.quote_resolved_json)
    inbound = inbound_for_pending_reply(session, tenant_id, reply)
    conversation_history = _conversation_history_for_pending_reply(session, tenant_id, reply)
    quote_pdf_url = (
        quote_pdf_dashboard_url(tenant_slug, reply.quote_external_id, inline=True)
        if reply.quote_external_id
        else None
    )
    editor_html = reply.draft_html
    if not editor_html and reply.channel == ConnectorType.EMAIL.value:
        editor_html = email_draft_html_from_markdown(reply.draft_text)
    manual_attachments, _quote_attachments = partition_attachment_entries(
        reply.attachments_json,
        settings=settings,
        tenant_slug=tenant_slug,
        reply_id=reply.id,
    )
    outbound_config = (
        _outbound_email_config(session, tenant_id)
        if reply.channel == ConnectorType.EMAIL.value
        else {}
    )
    outbound_subject = resolve_email_subject(
        draft_subject=coalesce_stored_email_subject(
            stored_draft_subject=reply.draft_subject,
            connector_config=outbound_config,
            inbound_subject=inbound.get("subject"),
        ),
        connector_config=outbound_config,
        inbound_subject=inbound.get("subject") or None,
    )
    thread_resolution_used_llm: bool | None = None
    thread_resolution_tooltip: str | None = None
    if reply.channel == ConnectorType.EMAIL.value:
        draft = inbound.get("mail_draft")
        if draft is None:
            draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_for_pending_reply(
                reply
            )
        audit = audit_from_json(draft.thread_resolution_json) if draft else None
        if audit is not None:
            thread_resolution_used_llm = audit.used_llm
            thread_resolution_tooltip = format_resolution_tooltip(audit)
    return {
        "reply": reply,
        "resolved_lines": resolved,
        "is_quote": reply.fulfillment_kind == FulfillmentKind.ERPNEXT_QUOTE,
        "has_erpnext_quote": bool(reply.quote_external_id),
        "manual_attachment_count": len(manual_attachments),
        "inbound_subject": inbound["subject"],
        "inbound_text": inbound["text"],
        "outbound_subject": outbound_subject,
        "conversation_history": conversation_history,
        "thread_resolution_used_llm": thread_resolution_used_llm,
        "thread_resolution_tooltip": thread_resolution_tooltip,
        "quote_pdf_url": quote_pdf_url,
        "erpnext_quote_url": _erpnext_quotation_edit_url(
            session, tenant_id, reply.quote_external_id
        ),
        "editor_html": editor_html,
        "blocked_from_addr": _email_from_addr_for_reply(reply)
        if reply.channel == ConnectorType.EMAIL.value
        else None,
    }


def _pending_replies_for_ui(
    session: Session,
    tenant_id: int,
    pending: list,
    *,
    tenant_slug: str,
    settings: Settings,
) -> list[dict]:
    return [
        _pending_reply_ui_item(
            session, tenant_id, reply, tenant_slug=tenant_slug, settings=settings
        )
        for reply in pending
    ]


def _pending_reply_for_tenant(
    pending_repo: SqlAlchemyPendingReplyRepository,
    reply_id: int,
    tenant_id: int,
) -> PendingReply:
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Reply not found")
    return reply


def _require_pending_email_reply(reply: PendingReply) -> None:
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    if reply.channel != ConnectorType.EMAIL.value:
        raise HTTPException(status_code=400, detail="Attachments are only supported for email")


def _persist_validation_email_draft_from_form(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    form,
    edited_by: str,
    outbound_config: dict | None = None,
) -> PendingReply:
    if reply.channel != ConnectorType.EMAIL.value:
        return reply
    if "draft_html" in form:
        try:
            reply = save_pending_reply_draft(
                session,
                tenant_id=tenant_id,
                reply=reply,
                draft_html=str(form.get("draft_html", "")),
                draft_subject=str(form.get("draft_subject", "")),
                edited_by=edited_by,
            )
        except DraftEditError:
            pass
    if outbound_config is not None:
        reply = persist_validation_email_subject(
            session,
            tenant_id=tenant_id,
            reply=reply,
            form_subject=str(form.get("draft_subject", "")),
            outbound_config=outbound_config,
        )
    return reply


def _merge_resolved_selection(form, reply) -> str | None:
    lines = resolved_lines_from_json(reply.quote_resolved_json)
    if not lines:
        return reply.quote_resolved_json
    updated: list[dict] = []
    for idx, line in enumerate(lines):
        selected = str(form.get(f"line_{idx}_item_code", "")).strip()
        row = dict(line)
        if selected:
            for cand in row.get("candidates") or []:
                if str(cand.get("item_code")) == selected:
                    row["item_code"] = cand.get("item_code")
                    row["item_name"] = cand.get("item_name")
                    row["rate"] = cand.get("rate")
                    row["uom"] = cand.get("uom")
                    row["status"] = "resolved"
                    row["match_score"] = cand.get("score", row.get("match_score"))
                    break
        updated.append(row)
    return json.dumps(updated, ensure_ascii=True)


async def _integration_config_from_request(
    integration_type: str,
    form,
    *,
    existing: Integration | None,
) -> dict:
    schema_fields = integration_fields_for(integration_type)
    field_values = {field.key: str(form.get(field.key, "")).strip() for field in schema_fields}
    for field in schema_fields:
        if field.input_type == "checkbox":
            field_values[field.key] = "on" if form.get(field.key) == "on" else ""
    incoming = _integration_config_from_form(integration_type, field_values)
    return _merge_integration_config(existing.config if existing else None, incoming)


def _integration_configs_for_client(integrations: list[Integration]) -> dict[str, dict]:
    secrets = secret_integration_keys()
    out: dict[str, dict] = {}
    for integration in integrations:
        cfg: dict = {}
        for key, value in integration.config.items():
            cfg[key] = "" if key in secrets else value
        connected = (
            is_quickbooks_connected(integration.config)
            if integration.type == IntegrationType.QUICKBOOKS
            else True
        )
        out[integration.type.value] = {
            "active": integration.active,
            "connected": connected,
            "config": cfg,
        }
    return out


def _merge_connector_config(existing: dict | None, incoming: dict) -> dict:
    secrets = secret_connector_keys()
    preserve_if_empty = secrets | oauth_managed_connector_keys() | frozenset({"process_since"})
    base = dict(existing or {})
    for key, value in incoming.items():
        if key in preserve_if_empty and not str(value).strip():
            continue
        if key == "process_since" and str(value).strip():
            parsed = parse_from_form(str(value))
            base[key] = parsed if parsed else base.get(key, "")
            continue
        base[key] = value
    return base


def _connector_configs_for_client(connectors: list[Connector]) -> dict[str, dict]:
    secrets = secret_connector_keys()
    out: dict[str, dict] = {}
    for connector in connectors:
        key = f"{connector.type.value}:{connector.direction.value}"
        cfg: dict = {}
        for field_key, value in connector.config.items():
            if field_key in secrets:
                cfg[field_key] = ""
            elif field_key == "process_since":
                cfg[field_key] = format_for_datetime_local(value)
            else:
                cfg[field_key] = value
        out[key] = {
            "active": connector.active,
            "mode": connector.mode.value,
            "config": cfg,
            "oauth_connected": is_oauth_connected(connector.config)
            or bool(str(connector.config.get("mail_connection_id", "")).strip()),
        }
    return out


def _connector_config_from_form(
    connector_type: str,
    direction: str,
    fields: dict[str, str],
    *,
    outbound_provider: str | None = None,
) -> dict:
    schema_fields = fields_for(
        connector_type, direction, outbound_provider=outbound_provider
    )
    secrets = secret_connector_keys()
    raw = {field.key: fields.get(field.key, "").strip() for field in schema_fields}
    cfg = {key: value for key, value in raw.items() if value or key not in secrets}
    for field in schema_fields:
        if field.input_type == "checkbox":
            cfg[field.key] = fields.get(field.key, "") == "on"
    return cfg


def _list_documents(settings: Settings, slug: str) -> list[str]:
    docs = tenant_docs_dir(settings, slug)
    return sorted(str(p.relative_to(docs)) for p in docs.rglob("*") if p.is_file())


def _parse_test_session_id(raw: str) -> str | None:
    session_id = raw.strip()
    if not session_id.startswith("test:") or len(session_id) <= 5:
        return None
    return session_id


_CLIENT_ADMIN_TEST_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def _client_admin_test_session_cookie_name(slug: str) -> str:
    return f"chatbot_test_anon_{slug}"


def _client_admin_test_session_cookie_path(slug: str) -> str:
    return f"/dashboard/bots/{slug}"


def _resolve_client_admin_test_session(
    request: Request,
    slug: str,
    *,
    query_session: str = "",
) -> tuple[str, str | None]:
    cookie_name = _client_admin_test_session_cookie_name(slug)
    stored = request.cookies.get(cookie_name, "").strip()
    session_id = stored if stored.startswith("test:") else ""
    parsed = _parse_test_session_id(query_session)
    if parsed:
        session_id = parsed
    if not session_id:
        session_id = f"test:{uuid.uuid4()}"
    cookie_to_set = session_id if stored != session_id else None
    return session_id, cookie_to_set


def _apply_test_chat_cookie(response, slug: str, cookie_value: str | None) -> None:
    path = _client_admin_test_session_cookie_path(slug)
    if cookie_value:
        response.set_cookie(
            _client_admin_test_session_cookie_name(slug),
            cookie_value,
            httponly=True,
            samesite="lax",
            max_age=_CLIENT_ADMIN_TEST_COOKIE_MAX_AGE,
            path=path,
        )


def _clear_client_admin_test_session_cookie(response, slug: str) -> None:
    response.delete_cookie(
        _client_admin_test_session_cookie_name(slug),
        path=_client_admin_test_session_cookie_path(slug),
    )


def _enforce_test_chat_params_for_user(
    user: User,
    user_service: UserService,
    request: Request,
    slug: str,
    *,
    test_email: str,
    test_phone: str,
    test_session: str,
    channel: str = "",
) -> tuple[str, str, str, str]:
    if user_service.can_use_full_test_chat(user):
        return test_email, test_phone, test_session, channel
    if test_email.strip() or test_phone.strip() or channel.strip():
        raise HTTPException(
            status_code=403,
            detail="Anonymous test chat only (admin required for email/phone/channel)",
        )
    if test_session.strip() and not test_session.strip().startswith("test:"):
        raise HTTPException(status_code=403, detail="Anonymous test chat only")
    session_id, _ = _resolve_client_admin_test_session(
        request,
        slug,
        query_session=test_session,
    )
    return "", "", session_id, ""


def _is_trackable_test_session(session_id: str) -> bool:
    return session_id.startswith(("email:", "whatsapp:", "test:"))


def _list_test_chat_sidebar_sessions(
    session: Session, tenant_id: int, *, limit: int = 50
) -> list[dict[str, str | None]]:
    """Sessions registered via dashboard Test chat (see test_chat_sessions)."""
    repo = TestChatSessionRepository(session, tenant_id)
    items: list[dict[str, str | None]] = []
    for row in repo.list_recent(limit=limit):
        user = row.created_by_user
        items.append(
            {
                "session_id": row.session_id,
                "user_email": user.email if user is not None else None,
            }
        )
    return items


def _dashboard_chat_session_id(
    user: User,
    *,
    test_email: str = "",
    test_phone: str = "",
    test_session: str = "",
    require_identity: bool = False,
    create_anonymous: bool = False,
) -> str:
    email, phone = resolve_manual_identity(test_email=test_email, test_phone=test_phone)
    channel_session = build_channel_session_id(email=email, phone=phone)
    if channel_session:
        return channel_session
    parsed = _parse_test_session_id(test_session)
    if parsed:
        return parsed
    if require_identity:
        raise HTTPException(
            status_code=400,
            detail="Test email or phone is required when a customer integration is active",
        )
    if create_anonymous:
        return f"test:{uuid.uuid4()}"
    return ""


def _has_customer_integration(session: Session, tenant_id: int) -> bool:
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    for itype in (IntegrationType.ERPNEXT, IntegrationType.QUICKBOOKS):
        if svc.find_active(tenant_id, type=itype):
            return True
    return False


def _first_active_outbound_connector(session: Session, tenant_id: int) -> Connector | None:
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    for conn in svc.list_for_tenant(tenant_id):
        if conn.direction == ConnectorDirection.OUT and conn.active:
            return conn
    return None


def _normalize_chat_test_channel(channel: str) -> str:
    return (channel or "").strip().lower()


def _is_simulated_chat_channel(channel: str) -> bool:
    normalized = _normalize_chat_test_channel(channel)
    return bool(normalized) and normalized != "auto"


def _outbound_connector_for_chat_test(
    session: Session, tenant_id: int, *, channel: str
) -> tuple[Connector | None, str | None]:
    if _is_simulated_chat_channel(channel):
        normalized = _normalize_chat_test_channel(channel)
        try:
            connector_type = ConnectorType(normalized)
        except ValueError:
            return None, f"Unknown channel: {normalized}"
        svc = ConnectorService(SqlAlchemyConnectorRepository(session))
        conn = get_outbound_connector(svc, tenant_id, connector_type)
        if conn is None:
            return None, f"No active outbound {normalized} connector is configured."
        return conn, None
    return _first_active_outbound_connector(session, tenant_id), None


def _apply_simulated_channel_outbound(
    session: Session,
    *,
    tenant: Tenant,
    slug: str,
    channel: str,
    session_id: str,
    test_email: str,
    test_phone: str,
    result,
    settings: Settings,
) -> tuple[str | None, bool, int | None, str | None]:
    """Mirror mail worker: always route through queue_after_chat for simulated channels."""
    connector, connector_error = _outbound_connector_for_chat_test(
        session, tenant.id, channel=channel
    )
    if connector is None:
        return connector_error, False, None, None
    email, phone = resolve_manual_identity(test_email=test_email, test_phone=test_phone)
    simulated = _normalize_chat_test_channel(channel)
    if simulated == ConnectorType.EMAIL.value and not email:
        return "Set Test email to simulate an email reply.", False, None, None
    recipient_id = email or phone or session_id
    queue_status, pending = queue_after_chat(
        session,
        tenant_id=tenant.id,
        connector=connector,
        session_id=session_id,
        recipient_id=recipient_id,
        result=result,
        settings=settings,
        tenant_slug=slug,
    )
    validation_url = f"/dashboard/bots/{slug}?tab=validation"
    if queue_status == "queued" and pending is not None:
        return f"Reply queued for validation ({simulated}).", True, pending.id, validation_url
    if queue_status == "ok":
        return f"Reply sent via {simulated} (connector direct mode).", False, None, None
    return "Reply was not queued.", False, None, None


def _run_dashboard_chat(
    request: Request,
    settings: Settings,
    tenant: Tenant,
    user: User,
    message: str,
    session: Session,
    *,
    test_email: str = "",
    test_phone: str = "",
    test_session: str = "",
):
    session_id = _dashboard_chat_session_id(
        user,
        test_email=test_email,
        test_phone=test_phone,
        test_session=test_session,
        require_identity=False,
        create_anonymous=True,
    )
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    hook_repo = SqlAlchemyHookEventRepository(session, tenant.id)
    chat = _build_chat_service(request, settings, tenant, repo, hook_repo, db_session=session)
    return session_id, chat.handle_user_message(session_id, message.strip())


def _chat_message_ui_dict(
    message: ChatMessage,
    *,
    session: Session | None = None,
    tenant_id: int | None = None,
    session_id: str | None = None,
) -> dict:
    context_debug = None
    if message.context_debug:
        dbg = message.context_debug
        context_debug = {
            "rag_chunks": dbg.rag_chunks,
            "rag_chars": dbg.rag_chars,
            "customer_chars": dbg.customer_chars,
            "system_chars": dbg.system_chars,
        }
    bubble: ValidationMessageBubble | None = None
    if session is not None and tenant_id is not None and session_id:
        bubble = bubble_for_session_message(
            session,
            tenant_id,
            session_id,
            message,
        )
    return bubble_to_ui_dict(
        bubble,
        role=message.role,
        content=message.content,
        context_debug=context_debug,
    )


class ChatTestSendOut(BaseModel):
    reply: str
    hook_type: str | None = None
    queued: bool = False
    validation_url: str | None = None
    pending_reply_id: int | None = None
    message: str | None = None
    quote_name: str | None = None
    pdf_url: str | None = None
    pdf_filename: str | None = None
    pdf_warning: str | None = None
    test_session: str | None = None
    context_size: dict[str, int] | None = None
    user_message: dict | None = None
    assistant_message: dict | None = None


class EmailTestSendOut(BaseModel):
    ok: bool
    message: str
    poll_hint_seconds: int


class EmailTestPollOut(BaseModel):
    ok: bool
    processed_mails: int
    message: str


class ValidationTranslateIn(BaseModel):
    target_lang: str
    draft_html: str
    draft_subject: str = ""


@router.get("/bots", response_class=HTMLResponse)
def bots_list(
    request: Request,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
):
    tenants = user_service.filter_tenants(user, tenant_service.list_tenants())
    if user_service.is_validation_only(user) and len(tenants) == 1:
        return RedirectResponse(url=_validation_inbox_url(tenants[0].slug), status_code=303)
    return templates.TemplateResponse(
        request,
        "bots/list.html",
        {"user": user, "tenants": tenants, "title": "Bots"},
    )


@router.get("/bots/new", response_class=HTMLResponse)
def bot_new_form(request: Request, user: User = Depends(require_admin)):
    return templates.TemplateResponse(
        request, "bots/new.html", {"user": user, "title": "New bot", "error": None}
    )


@router.post("/bots/new", response_class=HTMLResponse)
def bot_new_submit(
    request: Request,
    name: str = Form(...),
    slug: str = Form(""),
    prompt: str = Form("You are a helpful assistant."),
    user: User = Depends(require_admin),
    tenant_service: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
):
    result = tenant_service.create_tenant(name=name, slug=slug.strip() or None, prompt=prompt)
    session.commit()
    return templates.TemplateResponse(
        request,
        "bots/created.html",
        {"user": user, "tenant": result.tenant, "token": result.token, "title": "Bot created"},
    )


@router.get("/bots/import", response_class=HTMLResponse)
def bot_import_form(
    request: Request,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
):
    tenants = user_service.filter_tenants(user, tenant_service.list_tenants())
    return templates.TemplateResponse(
        request,
        "bots/import.html",
        {"user": user, "tenants": tenants, "title": "Import bot", "error": None},
    )


@router.post("/bots/import", response_class=HTMLResponse)
async def bot_import_submit(
    request: Request,
    mode: str = Form(...),
    bundle: UploadFile = File(...),
    new_name: str = Form(""),
    target_slug: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenants = user_service.filter_tenants(user, tenant_service.list_tenants())
    try:
        import_mode = ImportMode(mode)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "bots/import.html",
            {
                "user": user,
                "tenants": tenants,
                "title": "Import bot",
                "error": "Invalid import mode",
            },
            status_code=400,
        )
    if import_mode == ImportMode.OVERWRITE:
        if not target_slug.strip():
            return templates.TemplateResponse(
                request,
                "bots/import.html",
                {
                    "user": user,
                    "tenants": tenants,
                    "title": "Import bot",
                    "error": "Select a bot to overwrite",
                },
                status_code=400,
            )
        target = tenant_service.get_by_slug(target_slug.strip())
        if target is None or not user_service.can_access_tenant(user, target.id):
            return templates.TemplateResponse(
                request,
                "bots/import.html",
                {
                    "user": user,
                    "tenants": tenants,
                    "title": "Import bot",
                    "error": "Target bot not found or not accessible",
                },
                status_code=403,
            )
    raw = await bundle.read()
    try:
        result = import_bundle(
            raw,
            mode=import_mode,
            settings=settings,
            session=session,
            tenant_service=tenant_service,
            target_slug=target_slug.strip() or None,
            new_name=new_name.strip() or None,
        )
        session.commit()
    except BotBundleError as exc:
        return templates.TemplateResponse(
            request,
            "bots/import.html",
            {
                "user": user,
                "tenants": tenants,
                "title": "Import bot",
                "error": str(exc),
            },
            status_code=400,
        )
    if result.token:
        return templates.TemplateResponse(
            request,
            "bots/created.html",
            {
                "user": user,
                "tenant": result.tenant,
                "token": result.token,
                "title": "Bot imported",
            },
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{result.tenant.slug}?tab=config",
        status_code=303,
    )


@router.get("/users", response_class=HTMLResponse)
def users_list(
    request: Request,
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
):
    return templates.TemplateResponse(
        request,
        "users/list.html",
        {"user": user, "users": user_service.list_users(), "title": "Users"},
    )


@router.get("/users/new", response_class=HTMLResponse)
def user_new_form(request: Request, user: User = Depends(require_admin)):
    return templates.TemplateResponse(
        request,
        "users/new.html",
        {"user": user, "title": "New user", "error": None, "roles": UserRole},
    )


@router.post("/users/new")
def user_new_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    try:
        user_role = UserRole(role)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "users/new.html",
            {
                "user": user,
                "title": "New user",
                "error": "Invalid role",
                "roles": UserRole,
                "email": email,
            },
            status_code=400,
        )
    if user_service.find_by_email(email):
        return templates.TemplateResponse(
            request,
            "users/new.html",
            {
                "user": user,
                "title": "New user",
                "error": "Email already registered",
                "roles": UserRole,
                "email": email,
            },
            status_code=400,
        )
    user_service.create_user(email=email, password=password, role=user_role)
    session.commit()
    return RedirectResponse(url="/dashboard/users", status_code=303)


@router.get("/users/{user_id}", response_class=HTMLResponse)
def user_detail(
    request: Request,
    user_id: int,
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
    tenant_service: TenantService = Depends(get_tenant_service),
):
    target = user_service.get_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    allowed = set(user_service.tenant_ids_for_user(user_id))
    tenants = tenant_service.list_tenants()
    return templates.TemplateResponse(
        request,
        "users/detail.html",
        {
            "user": user,
            "target": target,
            "tenants": tenants,
            "allowed_tenant_ids": allowed,
            "roles": UserRole,
            "title": target.email,
        },
    )


@router.post("/users/{user_id}/role")
def user_set_role(
    user_id: int,
    role: str = Form(...),
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    try:
        user_role = UserRole(role)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role") from None
    if user_service.set_role(user_id, user_role) is None:
        raise HTTPException(status_code=404)
    session.commit()
    return RedirectResponse(url=f"/dashboard/users/{user_id}", status_code=303)


@router.post("/users/{user_id}/active")
def user_set_active(
    user_id: int,
    active: str = Form(""),
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    if user_service.set_active(user_id, active == "on") is None:
        raise HTTPException(status_code=404)
    session.commit()
    return RedirectResponse(url=f"/dashboard/users/{user_id}", status_code=303)


@router.post("/users/{user_id}/password")
def user_set_password(
    user_id: int,
    password: str = Form(...),
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    target = user_service.get_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404)
    user_service.set_password(target.email, password)
    session.commit()
    return RedirectResponse(url=f"/dashboard/users/{user_id}", status_code=303)


@router.post("/users/{user_id}/access")
async def user_set_access(
    user_id: int,
    request: Request,
    user: User = Depends(require_admin),
    user_service: UserService = Depends(get_user_service),
    tenant_service: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
):
    if user_service.get_by_id(user_id) is None:
        raise HTTPException(status_code=404)
    form = await request.form()
    selected = {int(k.removeprefix("tenant_")) for k in form if k.startswith("tenant_")}
    current = set(user_service.tenant_ids_for_user(user_id))
    for tid in selected - current:
        user_service.grant_access(user_id, tid)
    for tid in current - selected:
        user_service.revoke_access(user_id, tid)
    session.commit()
    return RedirectResponse(url=f"/dashboard/users/{user_id}", status_code=303)


@router.get("/bots/{slug}", response_class=HTMLResponse)
def bot_detail(
    request: Request,
    slug: str,
    tab: str = "config",
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    if user_service.is_validation_only(user) and tab != "validation":
        return RedirectResponse(url=_validation_inbox_url(slug), status_code=303)
    connectors = ConnectorService(SqlAlchemyConnectorRepository(session)).list_for_tenant(tenant.id)
    mail_conn_svc = MailConnectionService(session)
    mail_connections = mail_conn_svc.list_client_views(tenant.id)
    integrations = IntegrationService(SqlAlchemyIntegrationRepository(session)).list_for_tenant(
        tenant.id
    )
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    hooks = SqlAlchemyHookEventRepository(session, tenant.id).list_by_tenant(limit=50)
    ctx: dict = {
        "user": user,
        "tenant": tenant,
        "tab": tab,
        "connectors": connectors,
        "integrations": integrations,
        "hooks": hooks,
        "status_class": _status_class,
        "default_hook_instructions": DEFAULT_HOOK_INSTRUCTIONS,
        "can_edit": user_service.can_edit(user),
        "can_validate": user_service.can_validate(user),
        "validation_only": user_service.is_validation_only(user),
        "is_admin": user.role == UserRole.ADMIN,
        "title": tenant.name,
        "connector_types": ConnectorType,
        "connector_directions": ConnectorDirection,
        "connector_modes": ConnectorMode,
        "webhook_channels": _WEBHOOK_CHANNELS,
        "connector_schemas": connector_schemas_for_template(),
        "connector_schemas_json": json.dumps(connector_schemas_for_template()),
        "connector_configs_json": json.dumps(_connector_configs_for_client(connectors)),
        "process_since_default": format_for_datetime_local(process_since_now_iso()),
        "mail_connections": mail_connections,
        "mail_connections_json": json.dumps(
            [
                {
                    "id": mc.id,
                    "label": mc.label,
                    "provider": mc.provider,
                    "mailbox_email": mc.mailbox_email,
                    "active": mc.active,
                    "oauth_connected": mc.oauth_connected,
                    "config": mc.config,
                }
                for mc in mail_connections
            ]
        ),
        "connector_oauth_error": request.query_params.get("connector_oauth_error", "").strip(),
        "platform_microsoft_mail_oauth": platform_microsoft_mail_oauth_configured(settings),
        "platform_google_mail_oauth": platform_google_mail_oauth_configured(settings),
        "mail_oauth_callback_url": f"{_public_base_url(request, settings)}/dashboard/mail-oauth/callback",
        "integration_types": IntegrationType,
        "integration_schemas": integration_schemas_for_template(),
        "integration_meta": integration_meta_for_template(),
        "integration_endpoint": _integration_endpoint,
        "is_quickbooks_connected": is_quickbooks_connected,
        "integration_types_list": [t.value for t in IntegrationType],
        "pending_count": pending_repo.count_pending(tenant.id),
        "has_validation_connectors": any(c.mode == ConnectorMode.VALIDATION for c in connectors),
        "automation_modules_ui": _automation_modules_for_ui(integrations),
        "composed_hook_instructions": compose_hook_instructions(
            tenant,
            active_integrations=_active_integration_types(integrations),
        ),
        "erpnext_integration_active": IntegrationType.ERPNEXT.value
        in _active_integration_types(integrations),
        "has_customer_integration": bool(
            _active_integration_types(integrations)
            & {IntegrationType.ERPNEXT.value, IntegrationType.QUICKBOOKS.value}
        ),
        "has_email_in_connector": any(
            c.type == ConnectorType.EMAIL
            and c.direction == ConnectorDirection.IN
            and c.active
            for c in connectors
        ),
        "show_email_test_tab": settings.dev_mode
        and any(
            c.type == ConnectorType.EMAIL
            and c.direction == ConnectorDirection.IN
            and c.active
            for c in connectors
        ),
        "dev_mode": settings.dev_mode,
        "mail_poll_seconds": settings.mail_poll_seconds,
        "mailpit_web_url": settings.dev_mailpit_web_url if settings.dev_mode else None,
        "bot_dev_mode": tenant.config.dev_mode,
    }
    if tab == "documents":
        ctx["documents"] = _list_documents(settings, slug)
        ctx["sync_logs"] = request.session.pop("sync_logs", None)
    elif tab == "history":
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        ctx["sessions"] = repo.list_session_ids()
        sid = request.query_params.get("sid")
        if sid:
            ctx["selected_sid"] = sid
            messages = repo.list_messages(sid, limit=500)
            ctx["history_messages"] = messages
            ctx["history_bubbles"] = messages_to_bubbles(
                session, tenant.id, sid, messages
            )
    elif tab == "chat":
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        chat_full_test = user_service.can_use_full_test_chat(user)
        ctx["chat_full_test"] = chat_full_test
        if not chat_full_test:
            q_email = request.query_params.get("test_email", "").strip()
            q_phone = request.query_params.get("test_phone", "").strip()
            q_session = request.query_params.get("test_session", "").strip()
            if q_email or q_phone or (q_session and not q_session.startswith("test:")):
                return RedirectResponse(
                    url=f"/dashboard/bots/{slug}?tab=chat",
                    status_code=303,
                )
            test_email = ""
            test_phone = ""
            test_session, cookie_to_set = _resolve_client_admin_test_session(
                request, slug, query_session=q_session
            )
            if cookie_to_set:
                ctx["_chat_set_cookie"] = cookie_to_set
        else:
            test_email = request.query_params.get("test_email", "").strip()
            test_phone = request.query_params.get("test_phone", "").strip()
            test_session = request.query_params.get("test_session", "").strip()
        require_identity = ctx["has_customer_integration"] and chat_full_test
        test_sid = _dashboard_chat_session_id(
            user,
            test_email=test_email,
            test_phone=test_phone,
            test_session=test_session,
            require_identity=False,
        )
        ctx["chat_active_sid"] = test_sid
        ctx["chat_can_clear"] = bool(test_sid and test_sid.startswith("test:"))
        ctx["chat_session_id"] = test_sid or "(new anonymous session)"
        ctx["chat_test_email"] = test_email
        ctx["chat_test_phone"] = test_phone
        ctx["chat_test_session"] = test_session
        ctx["chat_require_identity"] = require_identity
        ctx["chat_validation_url"] = f"/dashboard/bots/{slug}?tab=validation"
        ctx["chat_dev_mode"] = tenant.config.dev_mode
        if test_sid and not test_sid.startswith("dashboard:"):
            messages = repo.list_messages(test_sid, limit=200)
        else:
            messages = []
        ctx["chat_messages_json"] = json.dumps(
            [
                _chat_message_ui_dict(m, session=session, tenant_id=tenant.id, session_id=test_sid)
                for m in messages
            ]
        )
        session_repo = TestChatSessionRepository(session, tenant.id)
        if chat_full_test:
            ctx["test_chat_sidebar_sessions"] = _list_test_chat_sidebar_sessions(
                session, tenant.id
            )
        else:
            ctx["test_chat_sidebar_sessions"] = []
        chat_quote_pdf: dict[str, str] | None = None
        session_row = session_repo.find(test_sid)
        if session_row and session_row.last_quote_name:
            pdf_path = quote_pdf_path(settings, slug, session_row.last_quote_name)
            if pdf_path is not None:
                safe_name = safe_quote_filename(session_row.last_quote_name)
                chat_quote_pdf = {
                    "quote_name": session_row.last_quote_name,
                    "pdf_url": quote_pdf_dashboard_url(slug, session_row.last_quote_name),
                    "pdf_filename": f"{safe_name}.pdf",
                    "message": f"Quotation: {session_row.last_quote_name}",
                }
        ctx["chat_quote_pdf_json"] = json.dumps(chat_quote_pdf) if chat_quote_pdf else "null"
        ctx["chat_outbound_connectors"] = [
            {"type": c.type.value, "mode": c.mode.value}
            for c in connectors
            if c.direction == ConnectorDirection.OUT and c.active
        ]
        ctx["show_chat_channel_selector"] = chat_full_test and bool(ctx["chat_outbound_connectors"])
    elif tab == "validation":
        vsub = request.query_params.get("vsub", "pending").strip() or "pending"
        if vsub not in ("pending", "approved", "rejected", "blacklist"):
            vsub = "pending"
        ctx["validation_subtab"] = vsub
        audit_svc = ValidationAuditService(session)
        ctx["validation_activity"] = audit_svc.list_activity(tenant.id, limit=VALIDATION_ACTIVITY_LIMIT)
        ctx["fulfillment_kind_quote"] = FulfillmentKind.ERPNEXT_QUOTE.value
        if vsub == "blacklist":
            ctx["blocked_senders"] = list(tenant.config.email_blocked_senders)
            replies = []
        elif vsub == "pending":
            replies = pending_repo.list_pending(tenant.id)
        elif vsub == "approved":
            replies = pending_repo.list_by_status(tenant.id, PendingReplyStatus.APPROVED)
        else:
            replies = pending_repo.list_by_status(tenant.id, PendingReplyStatus.REJECTED)
        ctx["validation_replies"] = replies
        ctx["pending_replies_ui"] = _pending_replies_for_ui(
            session, tenant.id, replies, tenant_slug=slug, settings=settings
        )
        ctx["pending_replies"] = replies
    elif tab == "integrations":
        by_type = {i.type.value: i for i in integrations}
        ctx["integrations_by_type"] = by_type
        ctx["multiple_active_integrations"] = sum(1 for i in integrations if i.active) > 1
        qb = by_type.get(IntegrationType.QUICKBOOKS.value)
        ctx["quickbooks_connected"] = is_quickbooks_connected(qb.config) if qb else False
        ctx["quickbooks_redirect_uri"] = _quickbooks_redirect_uri(request, settings, slug)
        requested = request.query_params.get("integration_type", "").strip()
        if requested in {t.value for t in IntegrationType}:
            selected = requested
        elif by_type:
            selected = next(iter(IntegrationType)).value
            for itype in IntegrationType:
                if itype.value in by_type:
                    selected = itype.value
                    break
        else:
            selected = IntegrationType.ERPNEXT.value
        ctx["selected_integration_type"] = selected
        ctx["integrations_config_json"] = json.dumps(_integration_configs_for_client(integrations))
        ctx["integration_schemas_json"] = json.dumps(integration_schemas_for_template())
        ctx["integration_meta_json"] = json.dumps(integration_meta_for_template())
    elif tab == "monitoring":
        if not user_service.can_edit(user):
            raise HTTPException(status_code=403)
        is_admin = user.role == UserRole.ADMIN
        monitoring_range = parse_monitoring_range(request.query_params)
        mon = MonitoringDashboardService(session, settings)
        ctx.update(
            mon.bot_context(
                tenant,
                since=monitoring_range.since,
                until=monitoring_range.until,
                is_admin=is_admin,
                usage_page=monitoring_range.usage_page,
            )
        )
        ctx["format_bytes"] = format_bytes
        ctx["format_count"] = format_count
        ctx["format_usd"] = format_usd
        ctx["is_admin"] = is_admin
        ctx["monitoring_query_string"] = monitoring_query_string
        ctx["client_billing_defaults"] = {
            "input": settings.client_billing_input_per_million_usd,
            "output": settings.client_billing_output_per_million_usd,
        }
    return _bot_detail_response(request, slug, ctx)


def _bot_detail_response(request: Request, slug: str, ctx: dict):
    cookie_value = ctx.pop("_chat_set_cookie", None)
    response = templates.TemplateResponse(request, "bots/detail.html", ctx)
    if cookie_value:
        _apply_test_chat_cookie(response, slug, cookie_value)
    return response


@router.post("/bots/{slug}/monitoring/client-billing")
def bot_save_client_billing(
    slug: str,
    client_billing_input: str = Form(""),
    client_billing_output: str = Form(""),
    usage_from: str = Form(""),
    usage_to: str = Form(""),
    usage_page: str = Form(""),
    user: User = Depends(require_admin),
    tenant_service: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)

    def _parse_rate(raw: str) -> Decimal | None:
        text = raw.strip()
        if not text:
            return None
        try:
            value = Decimal(text)
        except InvalidOperation:
            raise HTTPException(status_code=422, detail="Invalid billing rate") from None
        if not value.is_finite() or value < 0:
            raise HTTPException(
                status_code=422,
                detail="Billing rate must be a non-negative number",
            )
        return value

    tenant_service.update_client_billing(
        tenant.id,
        input_per_million_usd=_parse_rate(client_billing_input),
        output_per_million_usd=_parse_rate(client_billing_output),
    )
    session.commit()
    query_parts = ["tab=monitoring"]
    if usage_from.strip() and usage_to.strip():
        query_parts.append(f"usage_from={usage_from.strip()}")
        query_parts.append(f"usage_to={usage_to.strip()}")
    if usage_page.strip().isdigit() and int(usage_page.strip()) > 1:
        query_parts.append(f"usage_page={int(usage_page.strip())}")
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?{'&'.join(query_parts)}",
        status_code=303,
    )


@router.post("/bots/{slug}/settings")
def bot_save_settings(
    slug: str,
    name: str = Form(...),
    active: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    tenant_service.update_tenant(tenant.id, name=name.strip(), active=active == "on")
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


@router.post("/bots/{slug}/regenerate-token", response_class=HTMLResponse)
def bot_regenerate_token(
    request: Request,
    slug: str,
    user: User = Depends(require_admin),
    tenant_service: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    out = tenant_service.regenerate_token(tenant.id)
    if out is None:
        raise HTTPException(status_code=404)
    tenant, token = out
    session.commit()
    return templates.TemplateResponse(
        request,
        "bots/created.html",
        {"user": user, "tenant": tenant, "token": token, "title": "New API token"},
    )


@router.post("/bots/{slug}/delete")
def bot_delete(
    slug: str,
    confirm: str = Form(""),
    user: User = Depends(require_admin),
    tenant_service: TenantService = Depends(get_tenant_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    if confirm.strip() != slug:
        raise HTTPException(status_code=400, detail="Confirmation slug mismatch")
    tenant_service.delete_tenant(tenant.id, settings=settings)
    session.commit()
    return RedirectResponse(url="/dashboard/bots", status_code=303)


@router.post("/bots/{slug}/rag-config")
def bot_save_rag_config(
    slug: str,
    chat_model: str = Form("gemini-2.5-flash"),
    embedding_model: str = Form("gemini-embedding-001"),
    rewrite_model: str = Form("gemini-2.5-flash"),
    rag_enabled: str = Form(""),
    rag_rewrite_enabled: str = Form(""),
    rag_rewrite_lang_filter: str = Form(""),
    rag_top_k: int = Form(5),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(100),
    retrieval_language: str = Form("en"),
    dev_mode: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    existing = tenant.config
    cfg = TenantConfig(
        chat_model=chat_model.strip(),
        embedding_model=embedding_model.strip(),
        rewrite_model=rewrite_model.strip(),
        rag_enabled=rag_enabled == "on",
        rag_rewrite_enabled=rag_rewrite_enabled == "on",
        rag_rewrite_lang_filter=rag_rewrite_lang_filter == "on",
        rag_top_k=rag_top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_language=retrieval_language.strip(),
        dev_mode=dev_mode == "on",
        automation_modules=existing.automation_modules,
        hook_instructions_extra=existing.hook_instructions_extra,
    )
    tenant_service.update_tenant(tenant.id, config=cfg)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


@router.post("/bots/{slug}/config")
def bot_save_config(
    slug: str,
    prompt: str = Form(""),
    gemini_api_key: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    tenant_service.update_tenant(
        tenant.id,
        prompt=prompt,
        gemini_api_key=gemini_api_key.strip() or None,
        update_gemini_api_key=bool(gemini_api_key.strip()),
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


@router.post("/bots/{slug}/automation-config")
async def bot_save_automation_config(
    request: Request,
    slug: str,
    hook_instructions_extra: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    selected = [
        mod.id
        for mod in all_modules()
        if mod.ui_enabled and form.get(f"module_{mod.id}") == "on"
    ]
    cfg = tenant.config
    updated = TenantConfig(
        chat_model=cfg.chat_model,
        embedding_model=cfg.embedding_model,
        rewrite_model=cfg.rewrite_model,
        rag_enabled=cfg.rag_enabled,
        rag_rewrite_enabled=cfg.rag_rewrite_enabled,
        rag_rewrite_lang_filter=cfg.rag_rewrite_lang_filter,
        rag_top_k=cfg.rag_top_k,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        retrieval_language=cfg.retrieval_language,
        dev_mode=cfg.dev_mode,
        automation_modules=tuple(selected),
        hook_instructions_extra=hook_instructions_extra.strip(),
    )
    tenant_service.update_tenant(tenant.id, config=updated, hook_instructions=None, update_hook_instructions=True)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


@router.post("/bots/{slug}/automation-config/reset")
def bot_reset_automation_config(
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    cfg = tenant.config
    updated = TenantConfig(
        chat_model=cfg.chat_model,
        embedding_model=cfg.embedding_model,
        rewrite_model=cfg.rewrite_model,
        rag_enabled=cfg.rag_enabled,
        rag_rewrite_enabled=cfg.rag_rewrite_enabled,
        rag_rewrite_lang_filter=cfg.rag_rewrite_lang_filter,
        rag_top_k=cfg.rag_top_k,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        retrieval_language=cfg.retrieval_language,
        dev_mode=cfg.dev_mode,
        automation_modules=("core.orders",),
        hook_instructions_extra="",
        email_blocked_senders=cfg.email_blocked_senders,
    )
    tenant_service.update_tenant(tenant.id, config=updated, hook_instructions=None, update_hook_instructions=True)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


@router.post("/bots/{slug}/email-blacklist")
def bot_save_email_blacklist(
    slug: str,
    blocked_senders: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    addrs = [line.strip() for line in blocked_senders.splitlines() if line.strip()]
    tenant_service.update_blocked_senders(tenant.id, addrs)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


@router.post("/bots/{slug}/email-blacklist/unblock")
def bot_unblock_email_sender(
    slug: str,
    addr: str = Form(...),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    tenant_service.unblock_sender(tenant.id, addr)
    session.commit()
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=validation&vsub=blacklist",
        status_code=303,
    )


@router.post("/bots/{slug}/hooks/restore-default")
def bot_restore_hook_instructions(
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    return bot_reset_automation_config(
        slug,
        user=user,
        tenant_service=tenant_service,
        user_service=user_service,
        session=session,
    )


def _reconcile_tenant_documents(
    session: Session,
    *,
    settings: Settings,
    tenant: Tenant,
    fresh: bool = False,
) -> list[str]:
    merged = merge_tenant_settings(settings, tenant)
    docs = tenant_docs_dir(settings, tenant.slug)
    store = LanceVectorStore(settings.lancedb_root / tenant.slug)
    embedder = metered_embedder(
        inner=GeminiEmbedder(),
        tenant_id=tenant.id,
        operation="embed_ingest",
        model=merged.embedding_model,
        session=session,
    )
    sync_svc = IngestSyncService(
        settings=merged,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    return sync_svc.reconcile_root(docs, fresh=fresh)


@router.post("/bots/{slug}/documents")
async def bot_upload_documents(
    request: Request,
    slug: str,
    files: list[UploadFile] = File(...),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    docs = tenant_docs_dir(settings, slug)
    for f in files:
        name = Path(f.filename or "upload.bin").name
        dest = docs / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await f.read())
    request.session["sync_logs"] = _reconcile_tenant_documents(
        session, settings=settings, tenant=tenant, fresh=False
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=documents", status_code=303)


@router.post("/bots/{slug}/documents/delete")
def bot_delete_document(
    request: Request,
    slug: str,
    path: str = Form(...),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    docs = tenant_docs_dir(settings, slug)
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    target = (docs / rel).resolve()
    if not str(target).startswith(str(docs.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if target.is_file():
        target.unlink()
    request.session["sync_logs"] = _reconcile_tenant_documents(
        session, settings=settings, tenant=tenant, fresh=False
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=documents", status_code=303)


@router.post("/bots/{slug}/sync")
def bot_sync_documents(
    request: Request,
    slug: str,
    fresh: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    request.session["sync_logs"] = _reconcile_tenant_documents(
        session,
        settings=settings,
        tenant=tenant,
        fresh=fresh == "on",
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=documents", status_code=303)


@router.post("/bots/{slug}/connectors")
async def save_connector(
    request: Request,
    slug: str,
    connector_type: str = Form(...),
    direction: str = Form("in"),
    mode: str = Form("direct"),
    active: str = Form("on"),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    ctype, cdir, cmode, cfg, _outbound_provider = await _connector_config_from_request(
        form,
        tenant_id=tenant.id,
        session=session,
        connector_type=connector_type,
        direction=direction,
    )
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    svc.upsert(
        tenant_id=tenant.id,
        direction=cdir,
        type=ctype,
        mode=cmode,
        config=cfg,
        active=active == "on",
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


@router.post("/bots/{slug}/connectors/test")
async def test_connector_connection(
    request: Request,
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    connector_type = str(form.get("connector_type", "")).strip()
    direction = str(form.get("direction", "in")).strip()
    _ctype, _cdir, _cmode, cfg, _outbound_provider = await _connector_config_from_request(
        form,
        tenant_id=tenant.id,
        session=session,
        connector_type=connector_type,
        direction=direction,
    )
    result = run_connector_connection_test(
        connector_type, direction, cfg, session=session, tenant_id=tenant.id, settings=settings
    )
    return JSONResponse(result.to_dict())


def _connector_oauth_error_redirect(slug: str, message: str) -> RedirectResponse:
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=connectors&connector_oauth_error={url_quote(message)}",
        status_code=303,
    )


def _mail_oauth_connect(
    request: Request,
    slug: str,
    *,
    provider: str,
    direction: str,
    tenant,
    session: Session,
    settings: Settings,
) -> RedirectResponse:
    try:
        cdir = ConnectorDirection(direction)
    except ValueError:
        return _connector_oauth_error_redirect(slug, "Invalid connector direction.")
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    connector = svc.find(tenant.id, direction=cdir, type=ConnectorType.EMAIL)
    if connector is None:
        return _connector_oauth_error_redirect(
            slug, "Save the email connector first (Save connector), then click Connect."
        )
    cfg = connector.config
    auth_type = str(cfg.get("auth_type", EmailAuthType.PASSWORD.value)).strip()
    if provider == "microsoft" and auth_type != EmailAuthType.MICROSOFT_OAUTH.value:
        return _connector_oauth_error_redirect(
            slug,
            "Authentication must be Microsoft OAuth in the saved connector. "
            "Click Save connector after changing Authentication, then Connect again.",
        )
    if provider == "google" and auth_type != EmailAuthType.GOOGLE_OAUTH.value:
        return _connector_oauth_error_redirect(
            slug,
            "Authentication must be Google OAuth in the saved connector. "
            "Click Save connector after changing Authentication, then Connect again.",
        )
    if provider == "microsoft":
        client_id = str(cfg.get("microsoft_client_id", "")).strip()
        if not client_id or not str(cfg.get("microsoft_client_secret", "")).strip():
            return _connector_oauth_error_redirect(
                slug, "Microsoft client ID and secret are required in the saved connector."
            )
    else:
        client_id = str(cfg.get("google_client_id", "")).strip()
        if not client_id or not str(cfg.get("google_client_secret", "")).strip():
            return _connector_oauth_error_redirect(
                slug, "Google client ID and secret are required in the saved connector."
            )
    oauth_secret = _oauth_signing_secret(settings)
    state = sign_connector_oauth_state(
        slug=slug,
        direction=cdir.value,
        provider=provider,
        secret=oauth_secret,
    )
    redirect_uri = _mail_oauth_redirect_uri(request, settings, slug, provider)
    if provider == "microsoft":
        url = microsoft_build_authorize_url(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            direction=cdir.value,
        )
    else:
        url = google_build_authorize_url(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            direction=cdir.value,
        )
    return RedirectResponse(url=url, status_code=303)


def _mail_oauth_callback(
    request: Request,
    slug: str,
    *,
    provider: str,
    code: str,
    state: str,
    tenant,
    session: Session,
    settings: Settings,
) -> RedirectResponse:
    oauth_secret = _oauth_signing_secret(settings)
    try:
        state_data = verify_connector_oauth_state(state, secret=oauth_secret)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state") from exc
    if state_data["slug"] != slug or state_data["provider"] != provider:
        raise HTTPException(status_code=400, detail="OAuth state mismatch")
    if not code.strip():
        raise HTTPException(status_code=400, detail="Missing authorization code")
    try:
        cdir = ConnectorDirection(state_data["direction"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid direction in OAuth state") from exc
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    connector = svc.find(tenant.id, direction=cdir, type=ConnectorType.EMAIL)
    if connector is None:
        raise HTTPException(status_code=400, detail="Email connector not configured")
    cfg = dict(connector.config)
    redirect_uri = _mail_oauth_redirect_uri(request, settings, slug, provider)
    if provider == "microsoft":
        tokens = microsoft_exchange_code(
            code=code.strip(),
            client_id=str(cfg.get("microsoft_client_id", "")).strip(),
            client_secret=str(cfg.get("microsoft_client_secret", "")).strip(),
            redirect_uri=redirect_uri,
        )
    else:
        tokens = google_exchange_code(
            code=code.strip(),
            client_id=str(cfg.get("google_client_id", "")).strip(),
            client_secret=str(cfg.get("google_client_secret", "")).strip(),
            redirect_uri=redirect_uri,
        )
    cfg = apply_oauth_tokens_to_config(cfg, tokens)
    svc.upsert(
        tenant_id=tenant.id,
        direction=cdir,
        type=ConnectorType.EMAIL,
        mode=connector.mode,
        config=cfg,
        active=connector.active,
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


def _mail_connection_oauth_error_redirect(slug: str, message: str) -> RedirectResponse:
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=connectors&connector_oauth_error={url_quote(message)}",
        status_code=303,
    )


def _oauth_token_exchange_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                detail = str(body.get("error_description") or body.get("error") or "").strip()
                if detail:
                    return f"OAuth token exchange failed: {detail}"
        except Exception:
            pass
        return f"OAuth token exchange failed (HTTP {exc.response.status_code})."
    return f"OAuth failed: {exc}"


def _mail_connection_config_from_form(form) -> dict:
    provider = str(form.get("provider", "")).strip()
    incoming: dict = {}
    if provider == MailConnectionProvider.MICROSOFT_OAUTH.value:
        for key in ("microsoft_client_id", "microsoft_client_secret"):
            val = str(form.get(key, "")).strip()
            if val:
                incoming[key] = val
    elif provider == MailConnectionProvider.GOOGLE_OAUTH.value:
        for key in ("google_client_id", "google_client_secret"):
            val = str(form.get(key, "")).strip()
            if val:
                incoming[key] = val
    return incoming


@router.post("/bots/{slug}/mail-connections")
async def save_mail_connection(
    request: Request,
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    raw_id = str(form.get("connection_id", "")).strip()
    connection_id = int(raw_id) if raw_id else None
    label = str(form.get("label", "")).strip()
    provider = str(form.get("provider", "")).strip()
    mailbox_email = str(form.get("mailbox_email", "")).strip()
    active = str(form.get("active", "on")) == "on"
    config_incoming = _mail_connection_config_from_form(form)
    svc = MailConnectionService(session)
    try:
        connection = svc.upsert(
            tenant_id=tenant.id,
            connection_id=connection_id,
            label=label,
            provider=provider,
            mailbox_email=mailbox_email,
            config_incoming=config_incoming,
            active=active,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    if str(form.get("connect_after", "")).strip() == "1":
        return RedirectResponse(
            url=f"/dashboard/bots/{slug}/mail-connections/{connection.id}/connect",
            status_code=303,
        )
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


@router.post("/bots/{slug}/mail-connections/{connection_id}/delete")
async def delete_mail_connection(
    slug: str,
    connection_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = MailConnectionService(session)
    try:
        svc.delete(tenant.id, connection_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


@router.post("/bots/{slug}/mail-connections/{connection_id}/test")
async def test_mail_connection(
    request: Request,
    slug: str,
    connection_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    test = str(form.get("test", "imap")).strip().lower()
    connection = MailConnectionService(session).get_for_tenant(connection_id, tenant.id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Mail connection not found")
    if test == "preview":
        result = preview_mail_connection_inbox(
            connection,
            session=session,
            tenant_id=tenant.id,
            settings=settings,
        )
        return JSONResponse(result.to_dict())
    result = run_mail_connection_test(connection, test=test, session=session, settings=settings)
    return JSONResponse(result.to_dict())


@router.get("/bots/{slug}/mail-connections/{connection_id}/connect")
def mail_connection_connect(
    request: Request,
    slug: str,
    connection_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    connection = MailConnectionService(session).get_for_tenant(connection_id, tenant.id)
    if connection is None:
        return _mail_connection_oauth_error_redirect(slug, "Mail connection not found.")
    try:
        client_id, _client_secret = resolve_mail_oauth_credentials(connection, settings)
    except MailOAuthError:
        return _mail_connection_oauth_error_redirect(
            slug,
            "OAuth client ID and secret are required. Set platform credentials in .env "
            "or save them on the mail connection.",
        )
    provider = connection.provider.value
    oauth_secret = _oauth_signing_secret(settings)
    oauth_provider = "microsoft" if provider == MailConnectionProvider.MICROSOFT_OAUTH.value else "google"
    state = sign_mail_connection_oauth_state(
        slug=slug,
        connection_id=connection_id,
        provider=oauth_provider,
        secret=oauth_secret,
    )
    redirect_uri = _platform_mail_oauth_redirect_uri(request, settings)
    if oauth_provider == "microsoft":
        url = microsoft_build_authorize_url(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            for_connection=True,
        )
    else:
        url = google_build_authorize_url(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            direction="in",
        )
    return RedirectResponse(url=url, status_code=303)


def _process_mail_connection_oauth_callback(
    request: Request,
    *,
    code: str,
    state: str,
    error: str,
    error_description: str,
    error_subcode: str,
    settings: Settings,
    session: Session,
    user: User,
    user_service: UserService,
    tenant_service: TenantService,
    legacy_path_connection_id: int | None = None,
) -> RedirectResponse:
    oauth_secret = _oauth_signing_secret(settings)
    if error.strip():
        slug = _slug_from_mail_oauth_state(state, settings)
        if error == "access_denied" or error_subcode == "unauthorized_client":
            message = (
                "Your Microsoft admin has not approved this app. "
                "Ask your IT admin to grant admin consent for this application, then try Connect again."
            )
        else:
            detail = error_description.strip() or error.strip()
            message = f"OAuth authorization failed: {detail}"
        if slug:
            return _mail_connection_oauth_error_redirect(slug, message)
        return RedirectResponse(
            url=f"/dashboard?connector_oauth_error={url_quote(message)}",
            status_code=303,
        )

    try:
        state_data = verify_mail_connection_oauth_state(state, secret=oauth_secret)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state") from exc

    slug = str(state_data["slug"])
    connection_id = int(state_data["connection_id"])
    if legacy_path_connection_id is not None and legacy_path_connection_id != connection_id:
        logger.warning(
            "Mail OAuth legacy callback path connection_id=%s does not match state connection_id=%s",
            legacy_path_connection_id,
            connection_id,
        )

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)

    if not code.strip():
        raise HTTPException(status_code=400, detail="Missing authorization code")

    provider = str(state_data["provider"])
    svc = MailConnectionService(session)
    connection = svc.get_for_tenant(connection_id, tenant.id)
    if connection is None:
        raise HTTPException(status_code=400, detail="Mail connection not found")

    client_id, client_secret = connection_client_credentials(connection, settings)
    redirect_uri = _platform_mail_oauth_redirect_uri(request, settings)
    try:
        if provider == "microsoft":
            tokens = microsoft_exchange_code(
                code=code.strip(),
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
            )
        else:
            tokens = google_exchange_code(
                code=code.strip(),
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
            )
        svc.apply_oauth_tokens(connection, tokens)
        session.commit()
    except (MailOAuthError, MailConnectionError, httpx.HTTPError, ValueError) as exc:
        logger.exception(
            "Mail OAuth callback failed slug=%s connection_id=%s redirect_uri=%s",
            slug,
            connection_id,
            redirect_uri,
        )
        return _mail_connection_oauth_error_redirect(slug, _oauth_token_exchange_error_message(exc))
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


@router.get("/mail-oauth/callback")
def platform_mail_oauth_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    error_description: str = "",
    error_subcode: str = "",
    user: User = Depends(require_user),
    user_service: UserService = Depends(get_user_service),
    tenant_service: TenantService = Depends(get_tenant_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    return _process_mail_connection_oauth_callback(
        request,
        code=code,
        state=state,
        error=error,
        error_description=error_description,
        error_subcode=error_subcode,
        settings=settings,
        session=session,
        user=user,
        user_service=user_service,
        tenant_service=tenant_service,
    )


@router.get("/bots/{slug}/mail-connections/{connection_id}/callback")
def mail_connection_callback_legacy(
    request: Request,
    slug: str,
    connection_id: int,
    code: str = "",
    state: str = "",
    error: str = "",
    error_description: str = "",
    error_subcode: str = "",
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    _tenant_or_404(tenant_service, slug)
    return _process_mail_connection_oauth_callback(
        request,
        code=code,
        state=state,
        error=error,
        error_description=error_description,
        error_subcode=error_subcode,
        settings=settings,
        session=session,
        user=user,
        user_service=user_service,
        tenant_service=tenant_service,
        legacy_path_connection_id=connection_id,
    )


@router.get("/bots/{slug}/connectors/microsoft/connect")
def microsoft_mail_connect(
    request: Request,
    slug: str,
    direction: str = "in",
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    return _connector_oauth_error_redirect(
        slug,
        "Per-connector OAuth is deprecated. Create a Mail connection above, connect once, "
        "then select it on your email IN/OUT connectors.",
    )


@router.get("/bots/{slug}/connectors/microsoft/callback")
def microsoft_mail_callback(
    request: Request,
    slug: str,
    code: str = "",
    state: str = "",
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    return _mail_oauth_callback(
        request,
        slug,
        provider="microsoft",
        code=code,
        state=state,
        tenant=tenant,
        session=session,
        settings=settings,
    )


@router.get("/bots/{slug}/connectors/google/connect")
def google_mail_connect(
    request: Request,
    slug: str,
    direction: str = "in",
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    return _connector_oauth_error_redirect(
        slug,
        "Per-connector OAuth is deprecated. Create a Mail connection above, connect once, "
        "then select it on your email IN/OUT connectors.",
    )


@router.get("/bots/{slug}/connectors/google/callback")
def google_mail_callback(
    request: Request,
    slug: str,
    code: str = "",
    state: str = "",
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    return _mail_oauth_callback(
        request,
        slug,
        provider="google",
        code=code,
        state=state,
        tenant=tenant,
        session=session,
        settings=settings,
    )


@router.post("/bots/{slug}/integrations")
async def save_integration(
    request: Request,
    slug: str,
    integration_type: str = Form(...),
    active: str = Form("on"),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    try:
        itype = IntegrationType(integration_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid integration") from exc
    form = await request.form()
    schema_fields = integration_fields_for(integration_type)
    field_values = {
        field.key: str(form.get(field.key, "")).strip() for field in schema_fields
    }
    for field in schema_fields:
        if field.input_type == "checkbox":
            field_values[field.key] = "on" if form.get(field.key) == "on" else ""
    incoming = _integration_config_from_form(integration_type, field_values)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    existing = svc.find(tenant.id, type=itype)
    prev_enabled = (
        catalog_rag_effective_enabled(
            active=existing.active if existing else False,
            config=existing.config if existing else {},
        )
        if itype == IntegrationType.ERPNEXT
        else False
    )
    cfg = _merge_integration_config(existing.config if existing else None, incoming)
    saved = svc.upsert(tenant_id=tenant.id, type=itype, config=cfg, active=active == "on")
    if itype == IntegrationType.ERPNEXT:
        now_enabled = catalog_rag_effective_enabled(active=saved.active, config=saved.config)
        apply_catalog_rag_transition(
            session,
            settings,
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            integration_id=saved.id,
            config=saved.config,
            prev_enabled=prev_enabled,
            now_enabled=now_enabled,
            run_sync_background=_run_catalog_sync_background,
        )
    session.commit()
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=integrations&integration_type={integration_type}",
        status_code=303,
    )


@router.post("/bots/{slug}/integrations/{integration_id}/toggle")
def toggle_integration(
    slug: str,
    integration_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.get(integration_id)
    if integration is None or integration.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    prev_enabled = (
        catalog_rag_effective_enabled(active=integration.active, config=integration.config)
        if integration.type == IntegrationType.ERPNEXT
        else False
    )
    updated = svc.set_active(integration_id, not integration.active)
    if updated and updated.type == IntegrationType.ERPNEXT:
        now_enabled = catalog_rag_effective_enabled(active=updated.active, config=updated.config)
        apply_catalog_rag_transition(
            session,
            settings,
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            integration_id=updated.id,
            config=updated.config,
            prev_enabled=prev_enabled,
            now_enabled=now_enabled,
            run_sync_background=_run_catalog_sync_background,
        )
    session.commit()
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=integrations&integration_type={integration.type.value}",
        status_code=303,
    )


@router.post("/bots/{slug}/integrations/{integration_id}/delete")
def delete_integration(
    slug: str,
    integration_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.get(integration_id)
    if integration is None or integration.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if (
        integration.type == IntegrationType.ERPNEXT
        and catalog_rag_effective_enabled(active=integration.active, config=integration.config)
    ):
        apply_catalog_rag_transition(
            session,
            settings,
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            integration_id=integration.id,
            config=integration.config,
            prev_enabled=True,
            now_enabled=False,
            run_sync_background=_run_catalog_sync_background,
        )
    svc.delete(integration_id)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=integrations", status_code=303)


@router.post("/bots/{slug}/integrations/erpnext/create-customer")
async def create_erpnext_customer(
    request: Request,
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    test_email = str(form.get("test_email", "")).strip() or None
    test_phone = str(form.get("test_phone", "")).strip() or None
    customer_name = str(form.get("customer_name", "")).strip() or None
    company_name = str(form.get("company_name", "")).strip() or None
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    existing = svc.find(tenant.id, type=IntegrationType.ERPNEXT)
    cfg = await _integration_config_from_request(
        IntegrationType.ERPNEXT.value,
        form,
        existing=existing,
    )
    integration = svc.find_active(tenant.id, type=IntegrationType.ERPNEXT)
    if integration is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "Save and activate the ERPNext integration first.",
                "error": "integration_inactive",
                "customer": None,
                "created": False,
            }
        )
    client = ErpNextClient(cfg)
    result = create_erpnext_customer_for_test(
        client,
        cfg,
        test_email=test_email,
        test_phone=test_phone,
        customer_name=customer_name,
        company_name=company_name,
    )
    return JSONResponse(result)


@router.post("/bots/{slug}/integrations/erpnext/create-quotation")
async def create_erpnext_quotation(
    request: Request,
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    test_email = str(form.get("test_email", "")).strip() or None
    test_phone = str(form.get("test_phone", "")).strip() or None
    item_code = str(form.get("item_code", "")).strip()
    notes = str(form.get("notes", "")).strip() or None
    company_name = str(form.get("company_name", "")).strip() or None
    try:
        qty = int(str(form.get("qty", "1")).strip() or "1")
    except ValueError:
        qty = 0
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    existing = svc.find(tenant.id, type=IntegrationType.ERPNEXT)
    cfg = await _integration_config_from_request(
        IntegrationType.ERPNEXT.value,
        form,
        existing=existing,
    )
    integration = svc.find_active(tenant.id, type=IntegrationType.ERPNEXT)
    stream = str(form.get("stream", "")).strip() == "1"
    inactive_payload = {
        "ok": False,
        "message": "Save and activate the ERPNext integration first.",
        "error": "integration_inactive",
        "customer": None,
        "quote_name": None,
        "pdf_url": None,
    }
    if integration is None:
        if stream:

            async def inactive_stream():
                yield json.dumps({"event": "done", **inactive_payload}, ensure_ascii=False) + "\n"

            return StreamingResponse(inactive_stream(), media_type="application/x-ndjson")
        return JSONResponse(inactive_payload)

    if stream:

        async def quotation_stream():
            queue: Queue[dict] = Queue()

            def emit(message: str) -> None:
                queue.put({"event": "log", "message": message})

            progress = ProgressLog(emit=emit)

            def run() -> None:
                try:
                    client = ErpNextClient(cfg)
                    result = create_erpnext_quotation_for_test(
                        client,
                        cfg,
                        settings=settings,
                        tenant_slug=slug,
                        test_email=test_email,
                        test_phone=test_phone,
                        item_code=item_code,
                        qty=qty,
                        notes=notes,
                        company_name=company_name,
                        on_log=progress,
                    )
                    queue.put({"event": "done", **result})
                except Exception as exc:
                    queue.put(
                        {
                            "event": "done",
                            "ok": False,
                            "message": str(exc),
                            "error": "internal_error",
                            "customer": None,
                            "quote_name": None,
                            "pdf_url": None,
                        }
                    )

            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(None, run)
            while True:
                try:
                    while True:
                        item = queue.get_nowait()
                        yield json.dumps(item, ensure_ascii=False) + "\n"
                        if item.get("event") == "done":
                            await task
                            return
                except Empty:
                    pass
                if task.done() and queue.empty():
                    return
                await asyncio.sleep(0.05)

        return StreamingResponse(quotation_stream(), media_type="application/x-ndjson")

    client = ErpNextClient(cfg)
    result = create_erpnext_quotation_for_test(
        client,
        cfg,
        settings=settings,
        tenant_slug=slug,
        test_email=test_email,
        test_phone=test_phone,
        item_code=item_code,
        qty=qty,
        notes=notes,
        company_name=company_name,
    )
    return JSONResponse(result)


def _run_catalog_sync_background(
    settings: Settings,
    *,
    tenant_id: int,
    tenant_slug: str,
    integration_id: int,
    config: dict,
    force_rag_reconcile: bool = False,
) -> None:
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    try:
        with factory() as session:
            result = sync_erpnext_catalog_for_tenant(
                session,
                settings=settings,
                tenant_id=tenant_id,
                tenant_slug=tenant_slug,
                config=config,
                force_rag_reconcile=force_rag_reconcile,
            )
            update_catalog_sync_metadata(session, integration_id, result=result)
            session.commit()
            logger.info(
                "Catalog sync finished for %s: ok=%s message=%s",
                tenant_slug,
                result.ok,
                result.message,
            )
    except Exception:
        logger.exception("Catalog sync failed for %s", tenant_slug)
    finally:
        engine.dispose()


@router.post("/bots/{slug}/integrations/erpnext/sync-catalog")
async def sync_erpnext_catalog(
    slug: str,
    force_rag: str = Form(""),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    integration = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
        tenant.id,
        type=IntegrationType.ERPNEXT,
    )
    if integration is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "Save and activate the ERPNext integration first.",
                "error": "integration_inactive",
            }
        )
    config = dict(integration.config)
    thread = threading.Thread(
        target=_run_catalog_sync_background,
        kwargs={
            "settings": settings,
            "tenant_id": tenant.id,
            "tenant_slug": tenant.slug,
            "integration_id": integration.id,
            "config": config,
            "force_rag_reconcile": force_rag.strip().lower() in ("on", "1", "true", "yes"),
        },
        daemon=True,
    )
    thread.start()
    return JSONResponse(
        {
            "ok": True,
            "message": "Catalog sync started in background. Refresh this page to see status.",
        }
    )


@router.post("/bots/{slug}/integrations/erpnext/purge-catalog")
def purge_erpnext_catalog(
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    logs = purge_catalog_files_and_rag(
        session,
        settings=settings,
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
    )
    session.commit()
    return JSONResponse({"ok": True, "message": "Catalog files and RAG vectors purged.", "logs": logs})


@router.get("/bots/{slug}/integrations/erpnext/quotation-pdf/{quote_name}")
def download_test_quotation_pdf(
    slug: str,
    quote_name: str,
    inline: bool = False,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    safe = safe_quote_filename(quote_name)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid quotation name")
    path = quote_pdf_path(settings, slug, quote_name)
    if path is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return Response(
        content=path.read_bytes(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="{safe}.pdf"' if inline else f'attachment; filename="{safe}.pdf"'
            )
        },
    )


@router.post("/bots/{slug}/integrations/test")
async def test_integration(
    request: Request,
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    form = await request.form()
    integration_type = str(form.get("integration_type", "")).strip()
    test_email = str(form.get("test_email", "")).strip() or None
    test_phone = str(form.get("test_phone", "")).strip() or None
    try:
        itype = IntegrationType(integration_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid integration") from exc
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    existing = svc.find(tenant.id, type=itype)
    cfg = await _integration_config_from_request(integration_type, form, existing=existing)
    if itype == IntegrationType.QUICKBOOKS and not is_quickbooks_connected(cfg):
        return JSONResponse(
            {
                "ok": False,
                "message": "Connect QuickBooks via OAuth before testing.",
                "error": "not_connected",
                "customer": None,
                "orders": None,
                "quotations": None,
                "preview": None,
            }
        )
    result = run_integration_test(
        integration_type,
        cfg,
        test_email=test_email,
        test_phone=test_phone,
    )
    return JSONResponse(result.to_dict())


@router.get("/bots/{slug}/integrations/quickbooks/connect")
def quickbooks_connect(
    request: Request,
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.find(tenant.id, type=IntegrationType.QUICKBOOKS)
    if integration is None:
        raise HTTPException(status_code=400, detail="Save QuickBooks credentials first")
    cfg = integration.config
    client_id = str(cfg.get("client_id", "")).strip()
    client_secret = str(cfg.get("client_secret", "")).strip()
    if not client_id or not client_secret:
        raise HTTPException(status_code=400, detail="Client ID and Client Secret are required")
    oauth_secret = settings.session_secret.strip() or settings.app_secret_key.strip()
    if not oauth_secret:
        raise HTTPException(status_code=503, detail="SESSION_SECRET is required for OAuth")
    state = sign_oauth_state(slug=slug, secret=oauth_secret)
    url = build_authorize_url(
        client_id=client_id,
        redirect_uri=_quickbooks_redirect_uri(request, settings, slug),
        state=state,
        environment=str(cfg.get("environment", "sandbox")),
    )
    return RedirectResponse(url=url, status_code=303)


@router.get("/bots/{slug}/integrations/quickbooks/callback")
def quickbooks_callback(
    request: Request,
    slug: str,
    code: str = "",
    state: str = "",
    realmId: str = "",
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    oauth_secret = settings.session_secret.strip() or settings.app_secret_key.strip()
    try:
        state_slug = verify_oauth_state(state, secret=oauth_secret)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state") from exc
    if state_slug != slug:
        raise HTTPException(status_code=400, detail="OAuth state mismatch")
    if not code.strip():
        raise HTTPException(status_code=400, detail="Missing authorization code")
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.find(tenant.id, type=IntegrationType.QUICKBOOKS)
    if integration is None:
        raise HTTPException(status_code=400, detail="QuickBooks integration not configured")
    cfg = dict(integration.config)
    client_id = str(cfg.get("client_id", "")).strip()
    client_secret = str(cfg.get("client_secret", "")).strip()
    tokens = exchange_code(
        code=code.strip(),
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=_quickbooks_redirect_uri(request, settings, slug),
    )
    cfg["access_token"] = tokens.access_token
    cfg["refresh_token"] = tokens.refresh_token
    cfg["token_expires_at"] = tokens.expires_at
    cfg["realm_id"] = realmId.strip() or tokens.realm_id or cfg.get("realm_id", "")
    svc.upsert(
        tenant_id=tenant.id,
        type=IntegrationType.QUICKBOOKS,
        config=cfg,
        active=integration.active,
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=integrations", status_code=303)


@router.post("/bots/{slug}/integrations/quickbooks/disconnect")
def quickbooks_disconnect(
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.find(tenant.id, type=IntegrationType.QUICKBOOKS)
    if integration is None:
        raise HTTPException(status_code=404)
    cfg = dict(integration.config)
    for key in ("access_token", "refresh_token", "realm_id", "token_expires_at"):
        cfg.pop(key, None)
    svc.upsert(
        tenant_id=tenant.id,
        type=IntegrationType.QUICKBOOKS,
        config=cfg,
        active=integration.active,
    )
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=integrations", status_code=303)


@router.post("/bots/{slug}/connectors/{connector_id}/toggle")
def toggle_connector(
    slug: str,
    connector_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    conn = svc.get(connector_id)
    if conn is None or conn.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    svc.set_active(connector_id, not conn.active)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


@router.post("/bots/{slug}/connectors/{connector_id}/delete")
def delete_connector(
    slug: str,
    connector_id: int,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    conn = svc.get(connector_id)
    if conn is None or conn.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    svc.delete(connector_id)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=connectors", status_code=303)


@router.get("/bots/{slug}/validation/{reply_id}", response_class=HTMLResponse)
def validation_reply_detail(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = _pending_reply_for_tenant(pending_repo, reply_id, tenant.id)
    is_pending = reply.status == PendingReplyStatus.PENDING
    item = _pending_reply_ui_item(
        session, tenant.id, reply, tenant_slug=slug, settings=settings
    )
    attachment_rows = attachment_rows_for_ui(
        reply.attachments_json,
        settings=settings,
        tenant_slug=slug,
        reply_id=reply.id,
        quote_name=reply.quote_external_id,
    )
    quote_pdf_stale = _quote_pdf_stale_info(session, tenant.id, reply, tenant_slug=slug)
    reply_timeline = ValidationAuditService(session).list_timeline_for_reply(
        tenant.id, reply.id
    )
    return templates.TemplateResponse(
        request,
        "validation/detail.html",
        {
            "user": user,
            "tenant": tenant,
            "title": f"Validation #{reply.id} — {tenant.name}",
            "can_validate": user_service.can_validate(user) and is_pending,
            "is_pending": is_pending,
            "item": item,
            "reply": reply,
            "validation_inbox_url": _validation_inbox_url(slug),
            "attachment_rows": attachment_rows,
            "attachment_max_bytes": settings.attachment_max_bytes,
            "attachment_max_total_bytes": settings.attachment_max_total_bytes,
            "mailpit_web_url": settings.dev_mailpit_web_url if settings.dev_mode else None,
            "validation_error": request.session.pop("validation_error", None),
            "validation_warning": request.session.pop("validation_warning", None),
            "quote_pdf_stale": quote_pdf_stale,
            "reply_timeline": reply_timeline,
        },
    )


@router.post("/bots/{slug}/validation/{reply_id}/attachments")
async def upload_validation_attachments(
    slug: str,
    reply_id: int,
    files: list[UploadFile] = File(...),
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = _pending_reply_for_tenant(pending_repo, reply_id, tenant.id)
    _require_pending_email_reply(reply)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    attachments_json = reply.attachments_json
    new_entries: list[dict[str, str]] = []
    try:
        for upload in files:
            data = await upload.read()
            mime_type = validate_outbound_attachment_upload(
                settings,
                filename=upload.filename or "attachment",
                data=data,
                content_type=upload.content_type,
                existing_json=attachments_json,
            )
            entry = store_outbound_attachment(
                settings,
                slug,
                reply.id,
                upload.filename or "attachment",
                data,
                mime_type=mime_type,
            )
            new_entries.append(entry)
            attachments_json = merge_attachment_entries(attachments_json, [entry])
        pending_repo.update_quote_fields(reply.id, attachments_json=attachments_json)
        audit = ValidationAuditService(session)
        for entry in new_entries:
            audit.log_event(
                tenant_id=tenant.id,
                pending_reply_id=reply.id,
                action=ValidationAuditAction.ATTACHMENT_ADDED,
                actor_email=user.email,
                detail={"filename": entry.get("filename", "attachment")},
            )
        session.commit()
    except AttachmentValidationError as exc:
        for entry in new_entries:
            Path(entry["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rows = attachment_rows_for_ui(
        attachments_json,
        settings=settings,
        tenant_slug=slug,
        reply_id=reply.id,
        quote_name=reply.quote_external_id,
    )
    return JSONResponse({"ok": True, "attachments": rows})


@router.get("/bots/{slug}/validation/{reply_id}/attachments/file")
def view_validation_attachment_file(
    slug: str,
    reply_id: int,
    path: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = _pending_reply_for_tenant(pending_repo, reply_id, tenant.id)
    target = Path(path)
    allowed = is_user_attachment_path(settings, slug, reply.id, target)
    if not allowed and reply.quote_external_id:
        allowed = is_quote_pdf_path(settings, slug, reply.quote_external_id, target)
    if not allowed or not target.is_file():
        raise HTTPException(status_code=404, detail="Attachment not found")
    mime_type = "application/octet-stream"
    for entry in attachment_rows_for_ui(
        reply.attachments_json,
        settings=settings,
        tenant_slug=slug,
        reply_id=reply.id,
        quote_name=reply.quote_external_id,
    ):
        if entry.get("path") == path:
            mime_type = str(entry.get("mime_type") or mime_type)
            break
    inline = mime_type.startswith("image/") or mime_type == "application/pdf"
    disposition = "inline" if inline else "attachment"
    filename = target.name
    return Response(
        content=target.read_bytes(),
        media_type=mime_type,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@router.delete("/bots/{slug}/validation/{reply_id}/attachments")
def delete_validation_attachment(
    slug: str,
    reply_id: int,
    path: str,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = _pending_reply_for_tenant(pending_repo, reply_id, tenant.id)
    _require_pending_email_reply(reply)
    target = Path(path)
    if not is_user_attachment_path(settings, slug, reply.id, target):
        raise HTTPException(status_code=400, detail="Invalid attachment path")
    target.unlink(missing_ok=True)
    attachments_json = remove_attachment_entry(reply.attachments_json, path)
    pending_repo.update_quote_fields(reply.id, attachments_json=attachments_json)
    ValidationAuditService(session).log_event(
        tenant_id=tenant.id,
        pending_reply_id=reply.id,
        action=ValidationAuditAction.ATTACHMENT_REMOVED,
        actor_email=user.email,
        detail={"filename": target.name},
    )
    session.commit()
    rows = attachment_rows_for_ui(
        attachments_json,
        settings=settings,
        tenant_slug=slug,
        reply_id=reply.id,
        quote_name=reply.quote_external_id,
    )
    return JSONResponse({"ok": True, "attachments": rows})


@router.post("/bots/{slug}/validation/{reply_id}/approve")
async def approve_validation_reply(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    conn_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    connector = conn_svc.get(reply.connector_id)
    config = connector.config if connector else {}
    form = await request.form()
    if reply.channel == ConnectorType.EMAIL.value:
        reply = _persist_validation_email_draft_from_form(
            session,
            tenant_id=tenant.id,
            reply=reply,
            form=form,
            edited_by=user.email,
            outbound_config=_outbound_email_config(session, tenant.id),
        )
    resolved_json = _merge_resolved_selection(form, reply)
    if (
        reply.fulfillment_kind == FulfillmentKind.ERPNEXT_QUOTE
        and _quote_pdf_stale_info(session, tenant.id, reply, tenant_slug=slug).get("stale")
        and not form.get("confirm_stale_pdf")
    ):
        request.session["validation_warning"] = (
            "The quotation was updated in ERPNext since the PDF was last synced. "
            "Review the latest PDF, then use Proceed & send."
        )
        session.commit()
        return RedirectResponse(url=_validation_detail_url(slug, reply_id), status_code=303)
    fulfillment = QuoteFulfillmentService(session, settings=settings, tenant_slug=slug)
    try:
        fulfillment.fulfill_and_approve(
            reply,
            config=config,
            quote_resolved_json=resolved_json,
        )
    except QuoteFulfillmentError as exc:
        pending_repo.update_quote_fields(reply.id, fulfillment_error=str(exc))
        session.commit()
        request.session["validation_error"] = str(exc)
        return RedirectResponse(url=_validation_detail_url(slug, reply_id), status_code=303)
    fresh = pending_repo.find_by_id(reply_id)
    if fresh is not None:
        ValidationAuditService(session).resolve_reply(
            fresh,
            status=PendingReplyStatus.APPROVED,
            actor_email=user.email,
        )
    session.commit()
    return RedirectResponse(url=_validation_inbox_url(slug), status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/save")
async def save_validation_draft(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.channel != ConnectorType.EMAIL.value:
        raise HTTPException(status_code=400, detail="Only email drafts can be edited")
    form = await request.form()
    draft_html = str(form.get("draft_html", ""))
    draft_subject = str(form.get("draft_subject", ""))
    regenerated_from_raw = str(form.get("regenerated_from_raw", "")).strip() == "1"
    try:
        updated = save_pending_reply_draft(
            session,
            tenant_id=tenant.id,
            reply=reply,
            draft_html=draft_html,
            draft_subject=draft_subject,
            edited_by=user.email,
        )
        if regenerated_from_raw:
            ValidationAuditService(session).log_event(
                tenant_id=tenant.id,
                pending_reply_id=reply.id,
                action=ValidationAuditAction.REGENERATED,
                actor_email=user.email,
            )
        _ = updated
    except DraftEditError as exc:
        request.session["validation_error"] = str(exc)
    session.commit()
    return RedirectResponse(url=_validation_detail_url(slug, reply_id), status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/translate")
async def translate_validation_draft(
    slug: str,
    reply_id: int,
    body: ValidationTranslateIn,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.fulfillment_kind == FulfillmentKind.ERPNEXT_QUOTE:
        raise HTTPException(status_code=400, detail="Translation is not available for quote replies")
    try:
        translated = translate_pending_reply_draft(
            reply,
            draft_html=body.draft_html,
            draft_subject=body.draft_subject,
            target_lang=body.target_lang,
            tenant=tenant,
            settings=settings,
            session=session,
        )
    except ValidationTranslateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, **translated})


@router.post("/bots/{slug}/validation/{reply_id}/refresh-pdf")
def refresh_validation_quote_pdf(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.fulfillment_kind != FulfillmentKind.ERPNEXT_QUOTE:
        raise HTTPException(status_code=400, detail="Not a quote reply")
    if not reply.quote_external_id:
        raise HTTPException(status_code=400, detail="No quotation linked yet")
    try:
        attachments_json, erp_modified = refresh_quote_pdf(
            session,
            tenant_id=tenant.id,
            settings=settings,
            tenant_slug=slug,
            quote_name=reply.quote_external_id,
            existing_attachments_json=reply.attachments_json,
            reply_id=reply.id,
        )
        pending_repo.update_quote_fields(
            reply.id,
            attachments_json=attachments_json,
            fulfillment_error=None,
            quote_erp_modified=erp_modified,
        )
        ValidationAuditService(session).log_event(
            tenant_id=tenant.id,
            pending_reply_id=reply.id,
            action=ValidationAuditAction.REFRESH_PDF,
            actor_email=user.email,
            detail={"quote_name": reply.quote_external_id},
        )
        request.session.pop("validation_error", None)
        request.session.pop("validation_warning", None)
    except QuoteFulfillmentError as exc:
        pending_repo.update_quote_fields(reply.id, fulfillment_error=str(exc))
    session.commit()
    return RedirectResponse(url=_validation_detail_url(slug, reply_id), status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/resolve")
async def resolve_validation_quote(
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    from chatbot.adapters.erpnext.client import ErpNextClient
    from chatbot.application.outbound_orchestrator import _erpnext_client_for_tenant
    from chatbot.application.product_resolver import ProductResolver
    from chatbot.automation.modules.erpnext.quote import parse_quote_proposal

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.fulfillment_kind != FulfillmentKind.ERPNEXT_QUOTE:
        raise HTTPException(status_code=400, detail="Not a quote reply")
    client = _erpnext_client_for_tenant(session, tenant.id)
    if client is None:
        raise HTTPException(status_code=400, detail="ERPNext integration not active")
    if not reply.quote_proposal_json:
        raise HTTPException(status_code=400, detail="Missing quote proposal")
    payload = json.loads(reply.quote_proposal_json)
    proposal = parse_quote_proposal(payload) if isinstance(payload, dict) else None
    if proposal is None:
        raise HTTPException(status_code=400, detail="Invalid quote proposal")
    resolver = ProductResolver(client)
    lines = [
        {"product": line.product, "qty": line.qty, "item_code": line.item_code}
        for line in proposal.lines
    ]
    pending_repo.update_quote_fields(
        reply.id,
        quote_resolved_json=resolved_lines_to_json(resolver.resolve_all(lines)),
        fulfillment_error=None,
    )
    ValidationAuditService(session).log_event(
        tenant_id=tenant.id,
        pending_reply_id=reply.id,
        action=ValidationAuditAction.RESOLVE_PRODUCTS,
        actor_email=user.email,
    )
    session.commit()
    return RedirectResponse(url=_validation_detail_url(slug, reply_id), status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/retry-quote")
def retry_validation_quote(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    if reply.fulfillment_kind != FulfillmentKind.ERPNEXT_QUOTE:
        raise HTTPException(status_code=400, detail="Not a quote reply")
    if not reply.fulfillment_error:
        raise HTTPException(status_code=400, detail="No ERPNext error to retry")

    fulfillment = QuoteFulfillmentService(session, settings=settings, tenant_slug=slug)
    try:
        fulfillment.retry_quote_fulfillment(reply)
        fresh = pending_repo.find_by_id(reply_id)
        ValidationAuditService(session).log_event(
            tenant_id=tenant.id,
            pending_reply_id=reply.id,
            action=ValidationAuditAction.RETRY_QUOTE,
            actor_email=user.email,
            detail={"quote_name": fresh.quote_external_id if fresh else None},
        )
        request.session.pop("validation_error", None)
        request.session.pop("validation_warning", None)
    except QuoteFulfillmentError as exc:
        pending_repo.update_quote_fields(reply.id, fulfillment_error=str(exc))
        request.session["validation_error"] = str(exc)
    session.commit()
    return RedirectResponse(url=_validation_detail_url(slug, reply_id), status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/reject")
async def reject_validation_reply(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    form = await request.form()
    _reject_pending_reply(
        session,
        tenant_id=tenant.id,
        reply=reply,
        form=form,
        edited_by=user.email,
        settings=settings,
    )
    session.commit()
    return RedirectResponse(url=_validation_inbox_url(slug), status_code=303)


def _reject_pending_reply(
    session: Session,
    *,
    tenant_id: int,
    reply: PendingReply,
    form,
    edited_by: str,
    settings: Settings,
) -> None:
    _persist_validation_email_draft_from_form(
        session,
        tenant_id=tenant_id,
        reply=reply,
        form=form,
        edited_by=edited_by,
    )
    cleanup_pending_reply_attachments(reply)
    mark_imap_seen_for_pending_reply(
        session,
        tenant_id=tenant_id,
        reply=reply,
        settings=settings,
    )
    ValidationAuditService(session).resolve_reply(
        reply,
        status=PendingReplyStatus.REJECTED,
        actor_email=edited_by,
    )


@router.post("/bots/{slug}/validation/{reply_id}/regenerate")
async def regenerate_validation_reply(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    if reply.fulfillment_kind == FulfillmentKind.ERPNEXT_QUOTE:
        raise HTTPException(status_code=400, detail="Regenerate is not available for quote replies")
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    hook_repo = SqlAlchemyHookEventRepository(session, tenant.id)
    chat = _build_chat_service(request, settings, tenant, repo, hook_repo, db_session=session)
    try:
        generated = generate_pending_reply_from_raw(
            session,
            tenant,
            reply,
            settings=settings,
            chat=chat,
        )
    except ValidationRegenerateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {"ok": True, "draft_html": generated.draft_html, "draft_text": generated.draft_text}
    )


@router.post("/bots/{slug}/validation/{reply_id}/reject-blacklist")
async def reject_blacklist_validation_reply(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_validator),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    pending_repo = SqlAlchemyPendingReplyRepository(session)
    reply = pending_repo.find_by_id(reply_id)
    if reply is None or reply.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    blocked_addr = _email_from_addr_for_reply(reply)
    if not blocked_addr:
        blocked_addr = (reply.recipient_id or "").strip().lower()
    if not blocked_addr:
        raise HTTPException(status_code=400, detail="Cannot determine sender address")
    form = await request.form()
    _reject_pending_reply(
        session,
        tenant_id=tenant.id,
        reply=reply,
        form=form,
        edited_by=user.email,
        settings=settings,
    )
    tenant_service.add_blocked_sender(tenant.id, blocked_addr)
    ValidationAuditService(session).log_event(
        tenant_id=tenant.id,
        pending_reply_id=reply.id,
        action=ValidationAuditAction.REJECT_BLACKLIST,
        actor_email=user.email,
        detail={"blocked_addr": blocked_addr},
    )
    session.commit()
    request.session["validation_warning"] = f"Reply rejected and {blocked_addr} blocked."
    return RedirectResponse(url=_validation_inbox_url(slug), status_code=303)


@router.post("/bots/{slug}/chat-test/reset")
def bot_chat_test_reset(
    request: Request,
    slug: str,
    test_email: str = Form(""),
    test_phone: str = Form(""),
    test_session: str = Form(""),
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    test_email, test_phone, test_session, _ = _enforce_test_chat_params_for_user(
        user,
        user_service,
        request,
        slug,
        test_email=test_email,
        test_phone=test_phone,
        test_session=test_session,
    )
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    session_id = _dashboard_chat_session_id(
        user,
        test_email=test_email,
        test_phone=test_phone,
        test_session=test_session,
        require_identity=False,
    )
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing test session")
    if not session_id.startswith("test:"):
        raise HTTPException(
            status_code=403,
            detail="Only anonymous test sessions can be cleared here",
        )
    repo.clear_session(session_id)
    session_repo = TestChatSessionRepository(session, tenant.id)
    row = session_repo.find(session_id)
    if row and row.last_quote_name:
        pdf = quote_pdf_path(settings, slug, row.last_quote_name)
        if pdf is not None:
            try:
                pdf.unlink()
            except OSError:
                pass
    session_repo.clear_quote(session_id)
    session.commit()
    if user_service.can_use_full_test_chat(user):
        params = []
        if test_email.strip():
            params.append(f"test_email={test_email.strip()}")
        if test_phone.strip():
            params.append(f"test_phone={test_phone.strip()}")
        query = f"?tab=chat{'&' + '&'.join(params) if params else ''}"
    else:
        query = "?tab=chat"
    response = RedirectResponse(url=f"/dashboard/bots/{slug}{query}", status_code=303)
    if not user_service.can_use_full_test_chat(user):
        _clear_client_admin_test_session_cookie(response, slug)
    return response


@router.post("/bots/{slug}/chat-test/send", response_model=ChatTestSendOut)
def bot_chat_test_send(
    request: Request,
    response: Response,
    slug: str,
    message: str = Form(...),
    test_email: str = Form(""),
    test_phone: str = Form(""),
    test_session: str = Form(""),
    channel: str = Form(""),
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
) -> ChatTestSendOut:
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    test_email, test_phone, test_session, channel = _enforce_test_chat_params_for_user(
        user,
        user_service,
        request,
        slug,
        test_email=test_email,
        test_phone=test_phone,
        test_session=test_session,
        channel=channel,
    )
    try:
        session_id, result = _run_dashboard_chat(
            request,
            settings,
            tenant,
            user,
            message,
            session,
            test_email=test_email,
            test_phone=test_phone,
            test_session=test_session,
        )
        hook_type = getattr(result, "hook_type", None)
        queued = False
        pending_reply_id: int | None = None
        validation_url: str | None = None
        status_message: str | None = None
        quote_name: str | None = None
        pdf_url: str | None = None
        pdf_filename: str | None = None
        pdf_warning: str | None = None
        if _is_trackable_test_session(session_id):
            TestChatSessionRepository(session, tenant.id).upsert(
                session_id, created_by_user_id=user.id
            )
        if _is_simulated_chat_channel(channel):
            status_message, queued, pending_reply_id, validation_url = (
                _apply_simulated_channel_outbound(
                    session,
                    tenant=tenant,
                    slug=slug,
                    channel=channel,
                    session_id=session_id,
                    test_email=test_email,
                    test_phone=test_phone,
                    result=result,
                    settings=settings,
                )
            )
        elif hook_type:
            connector = _first_active_outbound_connector(session, tenant.id)
            if connector is None:
                status_message = (
                    "Hook detected but no active outbound connector is configured."
                )
            else:
                quote_created = False
                quote_hook = resolve_quote_hook(session, tenant.id, result)
                if quote_hook is not None:
                    proposal, resolved_json = quote_hook
                    from chatbot.application.customer_access_gate import can_create_quotation
                    from chatbot.application.outbound_orchestrator import erpnext_integration_for_tenant

                    integration = erpnext_integration_for_tenant(session, tenant.id)
                    if (
                        integration
                        and can_create_quotation(integration[1])
                        and all_lines_resolved(resolved_json)
                    ):
                        try:
                            created = create_quote_for_session(
                                session,
                                tenant_id=tenant.id,
                                settings=settings,
                                tenant_slug=tenant.slug,
                                session_id=session_id,
                                proposal=proposal,
                                resolved_json=resolved_json,
                                ttl_seconds=None,
                            )
                            quote_name = created.quote_name
                            pdf_url = created.pdf_url
                            pdf_filename = created.pdf_filename
                            pdf_warning = created.pdf_warning
                            status_message = f"Quotation created: {quote_name}"
                            quote_created = True
                            if _is_trackable_test_session(session_id):
                                TestChatSessionRepository(session, tenant.id).upsert(
                                    session_id,
                                    last_quote_name=quote_name,
                                    created_by_user_id=user.id,
                                )
                        except QuoteFulfillmentError as exc:
                            status_message = str(exc)
                if not quote_created:
                    email, phone = resolve_manual_identity(
                        test_email=test_email,
                        test_phone=test_phone,
                    )
                    recipient_id = email or phone or session_id
                    queue_status, pending = queue_after_chat(
                        session,
                        tenant_id=tenant.id,
                        connector=connector,
                        session_id=session_id,
                        recipient_id=recipient_id,
                        result=result,
                        settings=settings,
                        tenant_slug=tenant.slug,
                    )
                    if queue_status == "queued" and pending is not None:
                        queued = True
                        pending_reply_id = pending.id
                        validation_url = f"/dashboard/bots/{slug}?tab=validation"
                        if not status_message:
                            status_message = "Reply queued for validation."
                    elif not status_message:
                        status_message = "Hook detected but reply was not queued."
        session.commit()
        out_test_session = session_id if session_id.startswith("test:") else None
        context_size = None
        if getattr(result, "context_debug", None) is not None:
            dbg = result.context_debug
            context_size = {
                "rag_chunks": dbg.rag_chunks,
                "rag_chars": dbg.rag_chars,
                "customer_chars": dbg.customer_chars,
                "system_chars": dbg.system_chars,
            }
        conv_repo = SqlAlchemyConversationRepository(session, tenant.id)
        recent = conv_repo.list_messages(session_id, limit=2)
        user_message: dict | None = None
        assistant_message: dict | None = None
        if len(recent) >= 2:
            user_message = _chat_message_ui_dict(
                recent[0], session=session, tenant_id=tenant.id, session_id=session_id
            )
            assistant_message = _chat_message_ui_dict(
                recent[1], session=session, tenant_id=tenant.id, session_id=session_id
            )
        elif len(recent) == 1:
            assistant_message = _chat_message_ui_dict(
                recent[0], session=session, tenant_id=tenant.id, session_id=session_id
            )
        if (
            not user_service.can_use_full_test_chat(user)
            and session_id.startswith("test:")
            and request.cookies.get(_client_admin_test_session_cookie_name(slug), "").strip()
            != session_id
        ):
            _apply_test_chat_cookie(response, slug, session_id)
        return ChatTestSendOut(
            reply=result.text,
            hook_type=hook_type,
            queued=queued,
            validation_url=validation_url,
            pending_reply_id=pending_reply_id,
            message=status_message,
            quote_name=quote_name,
            pdf_url=pdf_url,
            pdf_filename=pdf_filename,
            pdf_warning=pdf_warning,
            test_session=out_test_session,
            context_size=context_size,
            user_message=user_message,
            assistant_message=assistant_message,
        )
    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _require_dev_email_test(settings: Settings) -> None:
    if not settings.dev_mode:
        raise HTTPException(status_code=403, detail="Email test is only available in DEV_MODE")


@router.post("/bots/{slug}/email-test/send", response_model=EmailTestSendOut)
def bot_email_test_send(
    slug: str,
    from_addr: str = Form("client@example.com"),
    subject: str = Form("Test email"),
    body: str = Form(...),
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
) -> EmailTestSendOut:
    _require_dev_email_test(settings)
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    try:
        config_in = get_email_test_connectors(session, tenant.id)
        inject_test_email(
            settings,
            config_in,
            from_addr=from_addr,
            subject=subject,
            body=body,
        )
    except EmailTestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    hint = max(1, settings.mail_poll_seconds)
    return EmailTestSendOut(
        ok=True,
        message="Mail injected into inbox. Use Process now or wait for the mail worker poll.",
        poll_hint_seconds=hint,
    )


@router.post("/bots/{slug}/email-test/poll", response_model=EmailTestPollOut)
def bot_email_test_poll(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
) -> EmailTestPollOut:
    _require_dev_email_test(settings)
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    reject_validation_only(user, user_service)
    factory = request.app.state.session_factory
    try:
        processed = poll_tenant_now(factory, settings, tenant)
    except EmailTestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    msg = (
        f"Processed {processed} mail(s)."
        if processed
        else "No new mail to process."
    )
    return EmailTestPollOut(ok=True, processed_mails=processed, message=msg)


@router.get("/bots/{slug}/export")
def bot_export(
    slug: str,
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    data = build_export(tenant, settings, session)
    filename = f"{slug}-export.zip"
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/hooks/{hook_id}/replay")
def replay_hook(
    hook_id: int,
    user: User = Depends(require_user),
    settings: Settings = Depends(get_settings_dep),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    if not settings.dev_mode or user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Hook replay is only available in DEV_MODE for platform admins",
        )
    repo = SqlAlchemyHookEventRepository(session, tenant_id=None)
    hooks = repo.list_by_tenant(limit=500)
    target = next((h for h in hooks if h.id == hook_id), None)
    if target is None:
        raise HTTPException(status_code=404)
    if not user_service.can_access_tenant(user, target.tenant_id):
        raise HTTPException(status_code=403)
    repo.reset_to_pending(hook_id)
    session.commit()
    tenant = tenant_service.get_by_id(target.tenant_id)
    slug = tenant.slug if tenant else ""
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=hooks", status_code=303)


@router.get("/monitoring", response_class=HTMLResponse)
def monitoring_global(
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
    tenant_service: TenantService = Depends(get_tenant_service),
    settings: Settings = Depends(get_settings_dep),
):
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403)
    monitoring_range = parse_monitoring_range(request.query_params)
    tenants = tenant_service.list_tenants()
    mon = MonitoringDashboardService(session, settings)
    payload = mon.global_context(
        tenants,
        since=monitoring_range.since,
        until=monitoring_range.until,
    )
    return templates.TemplateResponse(
        request,
        "monitoring/list.html",
        {
            "user": user,
            "rows": payload["rows"],
            "host_disk": payload["host_disk"],
            "usage_days": payload["usage_days"],
            "usage_from": payload["usage_from"],
            "usage_to": payload["usage_to"],
            "format_bytes": format_bytes,
            "format_count": format_count,
            "format_usd": format_usd,
            "usage_chart_json": payload["usage_chart_json"],
            "disk_bot_chart_json": payload["disk_bot_chart_json"],
            "disk_host_chart_json": payload["disk_host_chart_json"],
            "disk_pie_chart_json": payload["disk_pie_chart_json"],
            "platform_internal_cost_usd": payload["platform_internal_cost_usd"],
            "monitoring_query_string": monitoring_query_string,
            "title": "Monitoring",
        },
    )


@router.get("/hooks", response_class=HTMLResponse)
def hooks_global(
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403)
    hooks = SqlAlchemyHookEventRepository(session, tenant_id=None).list_by_tenant(limit=100)
    tenants = {t.id: t for t in tenant_service.list_tenants()}
    return templates.TemplateResponse(
        request,
        "hooks/list.html",
        {
            "user": user,
            "hooks": hooks,
            "tenants": tenants,
            "status_class": _status_class,
            "title": "Hooks",
            "dev_mode": settings.dev_mode,
        },
    )
