from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.hook_event_repository import SqlAlchemyHookEventRepository
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.bot_bundle_service import (
    BotBundleError,
    ImportMode,
    build_export,
    import_bundle,
)
from chatbot.application.channel_outbound import approve_pending_reply
from chatbot.application.hook_prompt_composer import compose_hook_instructions
from chatbot.application.product_resolver import resolved_lines_from_json, resolved_lines_to_json
from chatbot.application.quote_fulfillment_service import QuoteFulfillmentError, QuoteFulfillmentService
from chatbot.automation.modules.registry import all_modules
from chatbot.domain.models.fulfillment import FulfillmentKind
from chatbot.application.connector_service import ConnectorService
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
from chatbot.application.integration_test_service import run_integration_test
from chatbot.application.integration_service import IntegrationService
from chatbot.application.sync_service import IngestSyncService
from chatbot.application.tenant_service import TenantService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.application.user_service import UserService
from chatbot.config.settings import Settings
from chatbot.domain.constants import DEFAULT_HOOK_INSTRUCTIONS
from chatbot.domain.models.connector import Connector, ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.connector_schema import (
    EmailOutboundProvider,
    connector_schemas_for_template,
    fields_for,
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
from chatbot.domain.models.pending_reply import PendingReplyStatus
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
from chatbot.interfaces.web.deps import get_user_service, require_admin, require_editor, require_user
from chatbot.interfaces.web.templates import templates

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


def _inbound_for_pending_reply(session: Session, tenant_id: int, reply) -> dict[str, str]:
    subject = ""
    text = ""
    channel = (reply.channel or "").lower()
    if channel == "email":
        from_addr = (reply.recipient_id or reply.session_id.removeprefix("email:")).strip().lower()
        draft = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id).find_by_reply(
            from_addr, reply.draft_text
        )
        if draft:
            return {"subject": draft.subject, "text": draft.body_in}
    conv = SqlAlchemyConversationRepository(session, tenant_id)
    before = reply.created_at
    if before.tzinfo is None:
        before = before.replace(tzinfo=UTC)
    user_msg = conv.last_user_message_before(reply.session_id, before)
    if user_msg:
        text = user_msg
    return {"subject": subject, "text": text}


def _pending_replies_for_ui(session: Session, tenant_id: int, pending: list) -> list[dict]:
    out: list[dict] = []
    for reply in pending:
        resolved = resolved_lines_from_json(reply.quote_resolved_json)
        inbound = _inbound_for_pending_reply(session, tenant_id, reply)
        out.append(
            {
                "reply": reply,
                "resolved_lines": resolved,
                "is_quote": reply.fulfillment_kind == FulfillmentKind.ERPNEXT_QUOTE,
                "inbound_subject": inbound["subject"],
                "inbound_text": inbound["text"],
            }
        )
    return out


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
    preserve_if_empty = secrets | frozenset({"process_since"})
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


def _dashboard_chat_session_id(user: User) -> str:
    return f"dashboard:{user.id}"


def _run_dashboard_chat(
    request: Request,
    settings: Settings,
    tenant: Tenant,
    user: User,
    message: str,
    session: Session,
):
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    hook_repo = SqlAlchemyHookEventRepository(session, tenant.id)
    chat = _build_chat_service(request, settings, tenant, repo, hook_repo, db_session=session)
    return chat.handle_user_message(_dashboard_chat_session_id(user), message.strip())


class ChatTestSendOut(BaseModel):
    reply: str


class EmailTestSendOut(BaseModel):
    ok: bool
    message: str
    poll_hint_seconds: int


class EmailTestPollOut(BaseModel):
    ok: bool
    processed_mails: int
    message: str


@router.get("/bots", response_class=HTMLResponse)
def bots_list(
    request: Request,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
):
    tenants = user_service.filter_tenants(user, tenant_service.list_tenants())
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
    connectors = ConnectorService(SqlAlchemyConnectorRepository(session)).list_for_tenant(tenant.id)
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
        "is_admin": user.role == UserRole.ADMIN,
        "title": tenant.name,
        "connector_types": ConnectorType,
        "connector_directions": ConnectorDirection,
        "connector_modes": ConnectorMode,
        "webhook_channels": _WEBHOOK_CHANNELS,
        "connector_schemas": connector_schemas_for_template(),
        "connector_schemas_json": json.dumps(connector_schemas_for_template()),
        "connector_configs_json": json.dumps(_connector_configs_for_client(connectors)),
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
            ctx["history_messages"] = repo.list_messages(sid, limit=500)
    elif tab == "chat":
        repo = SqlAlchemyConversationRepository(session, tenant.id)
        test_sid = _dashboard_chat_session_id(user)
        ctx["chat_session_id"] = test_sid
        messages = repo.list_messages(test_sid, limit=200)
        ctx["chat_messages_json"] = json.dumps(
            [{"role": m.role.value, "content": m.content} for m in messages]
        )
    elif tab == "validation":
        ctx["pending_replies"] = pending_repo.list_pending(tenant.id)
        ctx["pending_replies_ui"] = _pending_replies_for_ui(session, tenant.id, ctx["pending_replies"])
        ctx["fulfillment_kind_quote"] = FulfillmentKind.ERPNEXT_QUOTE.value
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
    return templates.TemplateResponse(request, "bots/detail.html", ctx)


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
    )
    tenant_service.update_tenant(tenant.id, config=updated, hook_instructions=None, update_hook_instructions=True)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=config", status_code=303)


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
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=documents", status_code=303)


@router.post("/bots/{slug}/documents/delete")
def bot_delete_document(
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
    merged = merge_tenant_settings(settings, tenant)
    docs = tenant_docs_dir(settings, slug)
    store = LanceVectorStore(settings.lancedb_root / slug)
    embedder = GeminiEmbedder()
    sync_svc = IngestSyncService(
        settings=merged,
        embedder=embedder,
        vector_store=store,
        session=session,
        tenant_id=tenant.id,
    )
    request.session["sync_logs"] = sync_svc.reconcile_root(docs, fresh=fresh == "on")
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
    try:
        ctype = ConnectorType(connector_type)
        cdir = ConnectorDirection(direction)
        cmode = (
            ConnectorMode.DIRECT
            if cdir == ConnectorDirection.IN
            else ConnectorMode(mode)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid connector") from exc
    form = await request.form()
    outbound_provider: str | None = None
    if ctype == ConnectorType.EMAIL and cdir == ConnectorDirection.OUT:
        outbound_provider = (
            str(form.get("outbound_provider", EmailOutboundProvider.SMTP.value)).strip()
            or EmailOutboundProvider.SMTP.value
        )
    schema_fields = fields_for(
        connector_type, direction, outbound_provider=outbound_provider
    )
    field_values = {
        field.key: str(form.get(field.key, "")).strip() for field in schema_fields
    }
    for field in schema_fields:
        if field.input_type == "checkbox":
            field_values[field.key] = "on" if form.get(field.key) == "on" else ""
    incoming = _connector_config_from_form(
        connector_type, direction, field_values, outbound_provider=outbound_provider
    )
    svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    existing = svc.find(tenant.id, direction=cdir, type=ctype)
    cfg = _merge_connector_config(existing.config if existing else None, incoming)
    if ctype == ConnectorType.EMAIL and cdir == ConnectorDirection.IN:
        if not str(cfg.get("process_since", "")).strip():
            cfg["process_since"] = process_since_now_iso()
    if ctype == ConnectorType.EMAIL and cdir == ConnectorDirection.OUT:
        cfg["outbound_provider"] = outbound_provider or EmailOutboundProvider.SMTP.value
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


@router.post("/bots/{slug}/integrations")
async def save_integration(
    request: Request,
    slug: str,
    integration_type: str = Form(...),
    active: str = Form("on"),
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
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
    cfg = _merge_integration_config(existing.config if existing else None, incoming)
    svc.upsert(tenant_id=tenant.id, type=itype, config=cfg, active=active == "on")
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
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.get(integration_id)
    if integration is None or integration.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    svc.set_active(integration_id, not integration.active)
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
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    svc = IntegrationService(SqlAlchemyIntegrationRepository(session))
    integration = svc.get(integration_id)
    if integration is None or integration.tenant_id != tenant.id:
        raise HTTPException(status_code=404)
    svc.delete(integration_id)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=integrations", status_code=303)


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


@router.post("/bots/{slug}/validation/{reply_id}/approve")
async def approve_validation_reply(
    request: Request,
    slug: str,
    reply_id: int,
    user: User = Depends(require_editor),
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
    resolved_json = _merge_resolved_selection(form, reply)
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
        return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=validation", status_code=303)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=validation", status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/resolve")
async def resolve_validation_quote(
    slug: str,
    reply_id: int,
    user: User = Depends(require_editor),
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
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=validation", status_code=303)


@router.post("/bots/{slug}/validation/{reply_id}/reject")
def reject_validation_reply(
    slug: str,
    reply_id: int,
    user: User = Depends(require_editor),
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
    if reply.status != PendingReplyStatus.PENDING:
        raise HTTPException(status_code=400, detail="Reply is not pending")
    pending_repo.update_status(reply_id, PendingReplyStatus.REJECTED)
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=validation", status_code=303)


@router.post("/bots/{slug}/chat-test/reset")
def bot_chat_test_reset(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    repo = SqlAlchemyConversationRepository(session, tenant.id)
    repo.clear_session(_dashboard_chat_session_id(user))
    session.commit()
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=chat", status_code=303)


@router.post("/bots/{slug}/chat-test/send", response_model=ChatTestSendOut)
def bot_chat_test_send(
    request: Request,
    slug: str,
    message: str = Form(...),
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
) -> ChatTestSendOut:
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    try:
        result = _run_dashboard_chat(request, settings, tenant, user, message, session)
        session.commit()
        return ChatTestSendOut(reply=result.text)
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
    user: User = Depends(require_editor),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
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


@router.get("/hooks", response_class=HTMLResponse)
def hooks_global(
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
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
        },
    )
