from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
from chatbot.application.cac40_backtest_service import (
    default_ohlc_path,
    delete_run,
    get_run,
    list_runs,
    ohlc_info,
    read_ohlc_sync_status,
    read_ohlc_worker_status,
    resolve_chart_file,
    save_ohlc_upload,
    start_run,
    stop_run,
    sync_ohlc_from_ig,
)
from chatbot.application.connector_service import ConnectorService
from chatbot.application.integration_service import IntegrationService
from chatbot.application.tenant_service import TenantService
from chatbot.application.user_service import UserService
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.chart_renderer import normalize_pivot_period
from chatbot.config.settings import Settings
from chatbot.domain.models.integration import IntegrationType
from chatbot.domain.models.integration_schema import is_integration_allowed
from chatbot.domain.models.tenant import Tenant
from chatbot.domain.models.user import User
from chatbot.interfaces.api.deps import get_session, get_settings_dep, get_tenant_service
from chatbot.interfaces.web.deps import get_user_service, require_user
from chatbot.interfaces.web.templates import templates

router = APIRouter(prefix="/dashboard", tags=["cac40"])


def _tenant_or_404(tenant_service: TenantService, slug: str) -> Tenant:
    tenant = tenant_service.get_by_slug(slug)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return tenant


def _require_access(user: User, user_service: UserService, tenant: Tenant) -> None:
    if not user_service.can_access_tenant(user, tenant.id):
        raise HTTPException(status_code=403, detail="Forbidden")


def _require_cac40_active(tenant: Tenant, session: Session) -> None:
    if not is_integration_allowed(
        tenant.config.allowed_integrations, IntegrationType.CAC40_BACKTEST.value
    ):
        raise HTTPException(
            status_code=403,
            detail="CAC40 Backtest integration is not allowed for this bot",
        )
    active = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
        tenant.id, type=IntegrationType.CAC40_BACKTEST
    )
    if active is None:
        raise HTTPException(
            status_code=403,
            detail="CAC40 Backtest integration is not active. Save it as Active on the Integrations tab.",
        )


@router.get("/bots/{slug}/cac40", response_class=HTMLResponse)
def cac40_index(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    runs = list_runs(settings, slug)
    dataset = ohlc_info(settings, slug)
    ig_config = ConnectorService(SqlAlchemyConnectorRepository(session)).get_ig_config(
        tenant.id
    )
    integration = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
        tenant.id, type=IntegrationType.CAC40_BACKTEST
    )
    integ_cfg = dict(integration.config) if integration else {}
    defaults = Cac40Config().to_dict()
    defaults["symbol"] = str(integ_cfg.get("symbol") or defaults["symbol"] or "CAC40")
    defaults["epic"] = str(integ_cfg.get("epic") or defaults["epic"] or "IX.D.CAC.DAILY.IP")
    if integ_cfg.get("max_open_positions") not in (None, ""):
        try:
            defaults["max_open_positions"] = int(integ_cfg["max_open_positions"])
        except (TypeError, ValueError):
            pass
    return templates.TemplateResponse(
        request,
        "cac40/index.html",
        {
            "user": user,
            "tenant": tenant,
            "title": f"{defaults['symbol']} Backtest — {tenant.name}",
            "runs": runs,
            "ohlc": dataset,
            "ohlc_path": dataset["path"],
            "ohlc_exists": dataset["exists"],
            "bot_symbol": defaults["symbol"],
            "bot_epic": defaults["epic"],
            "default_config": defaults,
            "upload_error": request.query_params.get("upload_error"),
            "upload_ok": request.query_params.get("upload_ok"),
            "sync_error": request.query_params.get("sync_error"),
            "sync_ok": request.query_params.get("sync_ok"),
            "ig_connector_ready": bool(ig_config),
            "ohlc_sync_status": read_ohlc_sync_status(settings, slug),
            "ohlc_worker_status": read_ohlc_worker_status(settings),
            "cac40_ohlc_poll_seconds": settings.cac40_ohlc_poll_seconds,
        },
    )


@router.post("/bots/{slug}/cac40/ohlc")
async def cac40_upload_ohlc(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
    source: str = Form("evenor"),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    content = await file.read()
    try:
        info = save_ohlc_upload(
            settings,
            slug,
            filename=file.filename or "cac40_15m.csv",
            content=content,
            source=source,
        )
    except Exception as exc:
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/dashboard/bots/{slug}/cac40?upload_error={quote(str(exc))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}/cac40?upload_ok={info.get('bars', 0)}",
        status_code=303,
    )


@router.post("/bots/{slug}/cac40/ohlc/sync-ig")
def cac40_sync_ig(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    from urllib.parse import quote

    ig_config = ConnectorService(SqlAlchemyConnectorRepository(session)).get_ig_config(
        tenant.id
    )
    if not ig_config:
        return RedirectResponse(
            url=(
                f"/dashboard/bots/{slug}/cac40?"
                f"sync_error={quote('Configure an active IG connector first')}"
            ),
            status_code=303,
        )
    try:
        info = sync_ohlc_from_ig(
            settings,
            slug,
            ig_config=ig_config,
            allow_bootstrap=True,
            trigger="manual",
        )
    except Exception as exc:
        return RedirectResponse(
            url=f"/dashboard/bots/{slug}/cac40?sync_error={quote(str(exc))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}/cac40?sync_ok={info.get('added', 0)}",
        status_code=303,
    )


@router.post("/bots/{slug}/cac40/runs")
def cac40_start_run(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
    max_open_positions: int = Form(4),
    order_size: float = Form(1.0),
    spread_points: float = Form(1.5),
    llm_every_n: int = Form(6),
    llm_every_unit: str = Form("1h"),
    llm_mode: str = Form("live"),
    period: str = Form("1w"),
    lookback_15m: int = Form(96),
    lookback_1h: int = Form(72),
    lookback_1d: int = Form(60),
    warmup_bars: int = Form(14),
    chart_show_rsi: str = Form("1"),
    chart_show_pivots: str = Form("1"),
    chart_pivot_period: str = Form("D"),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    ohlc_path = default_ohlc_path(settings, slug)
    if not ohlc_path.exists():
        raise HTTPException(status_code=400, detail=f"OHLC file not found: {ohlc_path}")

    from chatbot.cac40.ohlc_store import BACKTEST_PERIODS

    period_key = (period or "1w").strip().lower()
    if period_key not in BACKTEST_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period. Choose: {', '.join(BACKTEST_PERIODS)}",
        )
    every_n, every_unit, every_bars = Cac40Config.llm_rate_from_form(
        every_n=llm_every_n, unit=llm_every_unit
    )
    integration = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
        tenant.id, type=IntegrationType.CAC40_BACKTEST
    )
    integ_cfg = dict(integration.config) if integration else {}
    symbol = str(integ_cfg.get("symbol") or "CAC40").strip() or "CAC40"
    epic = str(integ_cfg.get("epic") or "IX.D.CAC.DAILY.IP").strip() or "IX.D.CAC.DAILY.IP"

    cfg = Cac40Config(
        symbol=symbol,
        epic=epic,
        max_open_positions=max_open_positions,
        order_size=order_size,
        spread_points=spread_points,
        llm_every_n=every_n,
        llm_every_unit=every_unit,
        llm_every_bars=every_bars,
        llm_mode=llm_mode,
        backtest_period=period_key,
        lookback_15m=max(1, lookback_15m),
        lookback_1h=max(1, lookback_1h),
        lookback_1d=max(1, lookback_1d),
        warmup_bars=max(2, warmup_bars),
        chart_show_rsi=str(chart_show_rsi).strip().lower() in ("1", "true", "yes", "on"),
        chart_show_pivots=str(chart_show_pivots).strip().lower() in ("1", "true", "yes", "on"),
        chart_pivot_period=normalize_pivot_period(chart_pivot_period),
        gemini_model=(tenant.config.chat_model or settings.chat_model or "gemini-2.5-flash"),
        bot_id=tenant.slug,
    )
    from chatbot.interfaces.api.deps import _gemini_api_key

    api_key = _gemini_api_key(tenant, settings)
    run_id = start_run(
        settings,
        slug,
        config=cfg,
        ohlc_path=ohlc_path,
        api_key=api_key or "",
        tenant_id=tenant.id,
        session_factory=request.app.state.session_factory,
    )
    return RedirectResponse(url=f"/dashboard/bots/{slug}/cac40/runs/{run_id}", status_code=303)


@router.get("/bots/{slug}/cac40/runs/{run_id}", response_class=HTMLResponse)
def cac40_run_detail(
    request: Request,
    slug: str,
    run_id: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    try:
        run = get_run(settings, slug, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    return templates.TemplateResponse(
        request,
        "cac40/run.html",
        {
            "user": user,
            "tenant": tenant,
            "title": f"Run {run_id}",
            "run": run,
        },
    )


@router.post("/bots/{slug}/cac40/runs/{run_id}/stop")
def cac40_stop_run(
    slug: str,
    run_id: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    stop_run(run_id)
    return RedirectResponse(url=f"/dashboard/bots/{slug}/cac40/runs/{run_id}", status_code=303)


@router.get("/bots/{slug}/cac40/runs/{run_id}/charts/{chart_key}/{filename}")
def cac40_run_chart(
    slug: str,
    run_id: str,
    chart_key: str,
    filename: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    path = resolve_chart_file(settings, slug, run_id, chart_key, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(path, media_type="image/png", filename=filename)


@router.post("/bots/{slug}/cac40/runs/{run_id}/delete")
def cac40_delete_run(
    slug: str,
    run_id: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_cac40_active(tenant, session)
    if not delete_run(settings, slug, run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return RedirectResponse(url=f"/dashboard/bots/{slug}/cac40", status_code=303)
