from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.application.trader_backtest_service import (
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
from chatbot.application.trader_live_service import (
    adopt_ig_book,
    build_live_panel_snapshot,
    clear_live_history,
    clear_sync_log,
    get_live_report,
    load_live_config,
    preview_ig_book,
    read_sync_log,
    replay_live_decision,
    resolve_live_chart_file,
    run_live_cycle_now,
    save_live_config,
    set_live_mode,
)
from chatbot.application.connector_service import ConnectorService
from chatbot.application.tenant_service import TenantService
from chatbot.application.user_service import UserService
from chatbot.trader.config import TraderConfig
from chatbot.trader.chart_renderer import normalize_pivot_period
from chatbot.config.settings import Settings
from chatbot.domain.models.tenant import Tenant
from chatbot.domain.models.user import User
from chatbot.interfaces.api.deps import get_session, get_settings_dep, get_tenant_service
from chatbot.interfaces.web.deps import get_user_service, require_user
from chatbot.interfaces.web.templates import templates


def _bot_or_profile_point_value(integ_cfg: dict, profile) -> float:
    try:
        bot_pv = float(integ_cfg.get("point_value") or 0)
    except (TypeError, ValueError):
        bot_pv = 0.0
    if bot_pv > 0:
        return bot_pv
    return float(getattr(profile, "default_point_value", 1.0) or 1.0)

router = APIRouter(prefix="/dashboard", tags=["trader"])


def _tenant_or_404(tenant_service: TenantService, slug: str) -> Tenant:
    tenant = tenant_service.get_by_slug(slug)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return tenant


def _require_access(user: User, user_service: UserService, tenant: Tenant) -> None:
    if not user_service.can_access_tenant(user, tenant.id):
        raise HTTPException(status_code=403, detail="Forbidden")


def _require_trader_active(tenant: Tenant, session: Session) -> None:
    del session  # reserved for future session-scoped checks
    if not tenant.is_trader:
        raise HTTPException(
            status_code=403,
            detail="Trading is only available for trader bots",
        )


def _trader_integ_cfg(tenant: Tenant) -> dict:
    from chatbot.domain.trader_access import trader_settings_as_integration_dict

    return trader_settings_as_integration_dict(tenant)


@router.get("/bots/{slug}/trader", response_class=HTMLResponse)
def trader_index(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    session: Session = Depends(get_session),
):
    """Legacy URL — Trading now lives on the bot detail tab."""
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    ttab = request.query_params.get("ttab") or request.query_params.get("tab") or "live"
    if ttab not in ("data", "live", "backtest"):
        ttab = "live"
    # Preserve flash query params.
    qs = []
    for key in (
        "upload_error",
        "upload_ok",
        "sync_error",
        "sync_ok",
        "live_ok",
        "live_error",
    ):
        val = request.query_params.get(key)
        if val:
            from urllib.parse import quote

            qs.append(f"{key}={quote(val)}")
    extra = ("&" + "&".join(qs)) if qs else ""
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab={ttab}{extra}",
        status_code=303,
    )

@router.post("/bots/{slug}/trader/ohlc")
async def trader_upload_ohlc(
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
    _require_trader_active(tenant, session)
    content = await file.read()
    try:
        info = save_ohlc_upload(
            settings,
            slug,
            filename=file.filename or "ohlc_15m.csv",
            content=content,
            source=source,
        )
    except Exception as exc:
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/dashboard/bots/{slug}?tab=trading&ttab=data&upload_error={quote(str(exc))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab=data&upload_ok={info.get('bars', 0)}",
        status_code=303,
    )


@router.post("/bots/{slug}/trader/ohlc/sync-ig")
def trader_sync_ig(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    from urllib.parse import quote

    from chatbot.application.trader_live_service import resolve_primary_ig_config

    ig_config = resolve_primary_ig_config(
        settings, slug, session=session, tenant_id=tenant.id
    )
    if not ig_config:
        return RedirectResponse(
            url=(
                f"/dashboard/bots/{slug}?tab=trading&ttab=data&"
                f"sync_error={quote('Configure an active IG connector first')}"
            ),
            status_code=303,
        )
    try:
        bot_epic = str(tenant.config.trader.epic or "").strip() or None
        info = sync_ohlc_from_ig(
            settings,
            slug,
            ig_config=ig_config,
            allow_bootstrap=True,
            trigger="manual",
            epic=bot_epic,
        )
    except Exception as exc:
        return RedirectResponse(
            url=f"/dashboard/bots/{slug}?tab=trading&ttab=data&sync_error={quote(str(exc))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab=data&sync_ok={info.get('added', 0)}",
        status_code=303,
    )


@router.post("/bots/{slug}/trader/runs")
def trader_start_run(
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
    llm_temperature: float = Form(0.0),
    llm_trigger_mode: str = Form("levels"),
    llm_level_band_points: float = Form(15.0),
    allow_market_orders: str = Form("0"),
    prevent_loss_exits: str = Form("0"),
    flatten_before_close: str = Form("1"),
    flatten_lead_minutes: int = Form(30),
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
    _require_trader_active(tenant, session)
    ohlc_path = default_ohlc_path(settings, slug)
    if not ohlc_path.exists():
        raise HTTPException(status_code=400, detail=f"OHLC file not found: {ohlc_path}")

    from chatbot.trader.ohlc_store import BACKTEST_PERIODS

    period_key = (period or "1w").strip().lower()
    if period_key not in BACKTEST_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period. Choose: {', '.join(BACKTEST_PERIODS)}",
        )
    every_n, every_unit, every_bars = TraderConfig.llm_rate_from_form(
        every_n=llm_every_n, unit=llm_every_unit
    )
    trigger_mode = (llm_trigger_mode or "levels").strip().lower()
    if trigger_mode not in ("levels", "interval"):
        trigger_mode = "levels"
    band = max(0.1, float(llm_level_band_points or 15.0))
    temperature = max(0.0, min(1.0, float(llm_temperature if llm_temperature is not None else 0.0)))
    integ_cfg = _trader_integ_cfg(tenant)
    from chatbot.trader.profiles import get_profile

    profile = get_profile(integ_cfg.get("market_profile"))
    symbol = str(integ_cfg.get("symbol") or profile.default_symbol).strip() or profile.default_symbol
    epic = str(integ_cfg.get("epic") or profile.default_epic).strip() or profile.default_epic

    cfg = TraderConfig(
        symbol=symbol,
        epic=epic,
        max_open_positions=max_open_positions,
        order_size=order_size,
        spread_points=spread_points,
        llm_every_n=every_n,
        llm_every_unit=every_unit,
        llm_every_bars=every_bars,
        llm_mode=llm_mode,
        llm_temperature=temperature,
        llm_trigger_mode=trigger_mode,
        llm_level_band_points=band,
        allow_market_orders=str(allow_market_orders).strip().lower()
        in ("1", "true", "yes", "on"),
        prevent_loss_exits=str(prevent_loss_exits).strip().lower() in ("1", "true", "yes", "on"),
        flatten_before_close=str(flatten_before_close).strip().lower()
        in ("1", "true", "yes", "on"),
        flatten_lead_minutes=max(1, min(180, int(flatten_lead_minutes or 30))),
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
        system_prompt=str(tenant.prompt or ""),
        market_profile=profile.id,
        calendar_id=profile.calendar_id,
        point_value=_bot_or_profile_point_value(integ_cfg, profile),
        pnl_currency=str(integ_cfg.get("pnl_currency") or ""),
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
    return RedirectResponse(url=f"/dashboard/bots/{slug}/trader/runs/{run_id}", status_code=303)


@router.get("/bots/{slug}/trader/runs.json")
def trader_runs_json(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    """Lightweight run list for in-page polling (no full HTML refresh)."""
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    runs = list_runs(settings, slug)
    active = {"running", "stopping", "pending"}
    return JSONResponse(
        {
            "runs": runs,
            "has_active": any(str(r.get("status") or "") in active for r in runs),
        }
    )


@router.get("/bots/{slug}/trader/runs/{run_id}/status.json")
def trader_run_status_json(
    slug: str,
    run_id: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    """Backtest/live run progress for local polling without scroll jump."""
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    try:
        run = get_run(settings, slug, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    report = run.get("report") if isinstance(run.get("report"), dict) else {}
    decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
    return JSONResponse(
        {
            "run_id": run_id,
            "status": state.get("status"),
            "progress": state.get("progress"),
            "current_bar": state.get("current_bar"),
            "total_bars": state.get("total_bars"),
            "error": state.get("error"),
            "final_equity": report.get("final_equity"),
            "max_drawdown": report.get("max_drawdown"),
            "trades": report.get("trades"),
            "winrate": report.get("winrate"),
            "llm_calls_total": report.get("llm_calls_total"),
            "decisions_count": report.get("decisions_count")
            if report.get("decisions_count") is not None
            else len(decisions),
        }
    )


@router.get("/bots/{slug}/trader/runs/{run_id}", response_class=HTMLResponse)
def trader_run_detail(
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
    _require_trader_active(tenant, session)
    try:
        run = get_run(settings, slug, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    report_cfg = (run.get("report") or {}).get("config") if isinstance(run.get("report"), dict) else {}
    pnl_ccy = ""
    if isinstance(report_cfg, dict):
        pnl_ccy = str(report_cfg.get("pnl_currency") or "").strip().upper()
    if not pnl_ccy:
        pnl_ccy = str(tenant.config.trader.pnl_currency or "").strip().upper() or "USD"
    return templates.TemplateResponse(
        request,
        "trader/run.html",
        {
            "user": user,
            "tenant": tenant,
            "title": f"Run {run_id}",
            "run": run,
            "pnl_currency": pnl_ccy,
        },
    )


@router.post("/bots/{slug}/trader/runs/{run_id}/stop")
def trader_stop_run(
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
    _require_trader_active(tenant, session)
    stop_run(settings, slug, run_id)
    return RedirectResponse(url=f"/dashboard/bots/{slug}/trader/runs/{run_id}", status_code=303)


@router.get("/bots/{slug}/trader/runs/{run_id}/charts/{chart_key}/{filename}")
def trader_run_chart(
    slug: str,
    run_id: str,
    chart_key: str,
    filename: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
    download: int = 0,
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    path = resolve_chart_file(settings, slug, run_id, chart_key, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Chart not found")
    # Default: inline display for <img> / lightbox. ?download=1 forces attachment.
    if download:
        return FileResponse(path, media_type="image/png", filename=filename)
    return FileResponse(path, media_type="image/png")


@router.post("/bots/{slug}/trader/runs/{run_id}/delete")
def trader_delete_run(
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
    _require_trader_active(tenant, session)
    if not delete_run(settings, slug, run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return RedirectResponse(url=f"/dashboard/bots/{slug}?tab=trading&ttab=backtest", status_code=303)


def _strategy_from_form(
    *,
    max_open_positions: int,
    order_size: float,
    spread_points: float,
    llm_every_n: int,
    llm_every_unit: str,
    llm_temperature: float,
    llm_trigger_mode: str,
    llm_level_band_points: float,
    allow_market_orders: str,
    prevent_loss_exits: str,
    flatten_before_close: str,
    flatten_lead_minutes: int,
    lookback_15m: int,
    lookback_1h: int,
    lookback_1d: int,
    warmup_bars: int,
    chart_show_rsi: str,
    chart_show_pivots: str,
    chart_pivot_period: str,
) -> dict:
    every_n, every_unit, every_bars = TraderConfig.llm_rate_from_form(
        every_n=llm_every_n, unit=llm_every_unit
    )
    trigger_mode = (llm_trigger_mode or "levels").strip().lower()
    if trigger_mode not in ("levels", "interval"):
        trigger_mode = "levels"
    return {
        "max_open_positions": max_open_positions,
        "order_size": order_size,
        "spread_points": spread_points,
        "llm_every_n": every_n,
        "llm_every_unit": every_unit,
        "llm_every_bars": every_bars,
        "llm_temperature": max(0.0, min(1.0, float(llm_temperature if llm_temperature is not None else 0.0))),
        "llm_trigger_mode": trigger_mode,
        "llm_level_band_points": max(0.1, float(llm_level_band_points or 15.0)),
        "allow_market_orders": str(allow_market_orders).strip().lower()
        in ("1", "true", "yes", "on"),
        "prevent_loss_exits": str(prevent_loss_exits).strip().lower() in ("1", "true", "yes", "on"),
        "flatten_before_close": str(flatten_before_close).strip().lower()
        in ("1", "true", "yes", "on"),
        "flatten_lead_minutes": max(1, min(180, int(flatten_lead_minutes or 30))),
        "lookback_15m": max(1, lookback_15m),
        "lookback_1h": max(1, lookback_1h),
        "lookback_1d": max(1, lookback_1d),
        "warmup_bars": max(2, warmup_bars),
        "chart_show_rsi": str(chart_show_rsi).strip().lower() in ("1", "true", "yes", "on"),
        "chart_show_pivots": str(chart_show_pivots).strip().lower() in ("1", "true", "yes", "on"),
        "chart_pivot_period": normalize_pivot_period(chart_pivot_period),
    }


@router.post("/bots/{slug}/trader/live/config")
async def trader_live_save_config(
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
    llm_temperature: float = Form(0.0),
    llm_trigger_mode: str = Form("levels"),
    llm_level_band_points: float = Form(15.0),
    allow_market_orders: str = Form("0"),
    prevent_loss_exits: str = Form("0"),
    flatten_before_close: str = Form("1"),
    flatten_lead_minutes: int = Form(30),
    lookback_15m: int = Form(96),
    lookback_1h: int = Form(72),
    lookback_1d: int = Form(60),
    warmup_bars: int = Form(14),
    chart_show_rsi: str = Form("1"),
    chart_show_pivots: str = Form("1"),
    chart_pivot_period: str = Form("D"),
):
    from urllib.parse import quote

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    form = await request.form()
    ids = []
    for raw in form.getlist("ig_connector_id"):
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    strategy = _strategy_from_form(
        max_open_positions=max_open_positions,
        order_size=order_size,
        spread_points=spread_points,
        llm_every_n=llm_every_n,
        llm_every_unit=llm_every_unit,
        llm_temperature=llm_temperature,
        llm_trigger_mode=llm_trigger_mode,
        llm_level_band_points=llm_level_band_points,
        allow_market_orders=allow_market_orders,
        prevent_loss_exits=prevent_loss_exits,
        flatten_before_close=flatten_before_close,
        flatten_lead_minutes=flatten_lead_minutes,
        lookback_15m=lookback_15m,
        lookback_1h=lookback_1h,
        lookback_1d=lookback_1d,
        warmup_bars=warmup_bars,
        chart_show_rsi=chart_show_rsi,
        chart_show_pivots=chart_show_pivots,
        chart_pivot_period=chart_pivot_period,
    )
    current = load_live_config(settings, slug)
    save_live_config(
        settings,
        slug,
        {"mode": current["mode"], "ig_connector_ids": ids, "strategy": strategy},
    )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_ok={quote('Live config saved')}",
        status_code=303,
    )


@router.post("/bots/{slug}/trader/live/mode")
async def trader_live_set_mode(
    request: Request,
    slug: str,
    mode: str = Form(...),
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    from urllib.parse import quote

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    form = await request.form()
    ids: list[int] | None = None
    if "ig_connector_id" in form:
        ids = []
        for raw in form.getlist("ig_connector_id"):
            try:
                ids.append(int(raw))
            except (TypeError, ValueError):
                continue
    try:
        set_live_mode(
            settings,
            slug,
            mode,
            session=session,
            tenant_id=tenant.id,
            ig_connector_ids=ids,
        )
    except ValueError as exc:
        return RedirectResponse(
            url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_error={quote(str(exc))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_ok={quote('Mode set to ' + mode)}",
        status_code=303,
    )


@router.get("/bots/{slug}/trader/live/report", response_class=HTMLResponse)
def trader_live_report(
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
    _require_trader_active(tenant, session)
    run = get_live_report(settings, slug)
    return templates.TemplateResponse(
        request,
        "trader/run.html",
        {
            "user": user,
            "tenant": tenant,
            "title": f"Live results — {tenant.name}",
            "run": run,
            "live": True,
            "dev_mode": settings.dev_mode,
            "live_ok": request.query_params.get("live_ok"),
            "live_error": request.query_params.get("live_error"),
            "pnl_currency": str(tenant.config.trader.pnl_currency or "").strip().upper()
            or "USD",
        },
    )


@router.get("/bots/{slug}/trader/live/snapshot.json")
def trader_live_snapshot_json(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    """Open book + latest cycles for Trading → Live in-page polling."""
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    snap = build_live_panel_snapshot(settings, slug, cycle_limit=3)
    live_mode = str(snap.get("mode") or "off")
    sync_href = f"/dashboard/bots/{slug}/trader/live/book-sync"
    book_html = templates.env.from_string(
        "{% from 'trader/_open_book.html' import open_book_table %}"
        "{{ open_book_table(book, live_mode=live_mode, heading='Open book',"
        " embedded=true, sync_href=sync_href) }}"
    ).render(
        book=snap.get("book"),
        live_mode=live_mode,
        sync_href=sync_href,
    )
    cycles_html = templates.env.from_string(
        "{% from 'trader/_live_poll_parts.html' import live_cycles_list %}"
        "{{ live_cycles_list(cycles, slug) }}"
    ).render(cycles=snap.get("cycles") or [], slug=slug)
    return JSONResponse(
        {
            "mode": live_mode,
            "as_of": snap.get("as_of"),
            "book_as_of": snap.get("book_as_of"),
            "last_cycle_at": snap.get("last_cycle_at"),
            "fingerprint": snap.get("fingerprint"),
            "book_html": book_html,
            "cycles_html": cycles_html,
        }
    )


@router.get("/bots/{slug}/trader/live/charts/{cycle}/{filename}")
def trader_live_chart(
    slug: str,
    cycle: str,
    filename: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
    download: int = 0,
):
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    path = resolve_live_chart_file(settings, slug, cycle, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Chart not found")
    if download:
        return FileResponse(path, media_type="image/png", filename=filename)
    return FileResponse(path, media_type="image/png")


@router.post("/bots/{slug}/trader/live/run-once")
async def trader_live_run_once(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    from urllib.parse import quote

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    form = await request.form()
    # Checkbox: present "1" when checked (default in UI). Absent when unchecked.
    force_llm = str(form.get("force_llm") or "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )
    result = run_live_cycle_now(
        session,
        settings,
        slug,
        session_factory=getattr(request.app.state, "session_factory", None),
        force_llm=force_llm,
    )
    if result.get("ok"):
        return RedirectResponse(
            url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_ok={quote(str(result['message']))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_error={quote(str(result['message']))}",
        status_code=303,
    )


@router.post("/bots/{slug}/trader/live/replay-decision")
async def trader_live_replay_decision(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    """Dev-only: re-apply a stored LLM decision without calling Gemini."""
    from urllib.parse import quote

    if not settings.dev_mode:
        raise HTTPException(
            status_code=403,
            detail="Decision replay is only available when DEV_MODE=true",
        )
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    form = await request.form()
    cycle_dir = str(form.get("cycle_dir") or "").strip() or None
    result = replay_live_decision(
        session,
        settings,
        slug,
        cycle_dir=cycle_dir,
        session_factory=getattr(request.app.state, "session_factory", None),
    )
    # Prefer staying on report when replaying a specific cycle.
    redirect_base = (
        f"/dashboard/bots/{slug}/trader/live/report"
        if cycle_dir
        else f"/dashboard/bots/{slug}?tab=trading&ttab=live"
    )
    sep = "&" if "?" in redirect_base else "?"
    if result.get("ok"):
        return RedirectResponse(
            url=f"{redirect_base}{sep}live_ok={quote(str(result['message']))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"{redirect_base}{sep}live_error={quote(str(result['message']))}",
        status_code=303,
    )


@router.get("/bots/{slug}/trader/live/book-sync", response_class=HTMLResponse)
def trader_live_book_sync(
    request: Request,
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    """Preview local vs IG open book; apply via POST sync-book."""
    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    live_cfg = load_live_config(settings, slug)
    preview = preview_ig_book(session, settings, slug)
    return templates.TemplateResponse(
        request,
        "trader/book_sync.html",
        {
            "user": user,
            "tenant": tenant,
            "title": f"Book vs IG — {tenant.name}",
            "live_mode": live_cfg.get("mode") or "off",
            "preview": preview,
            "sync_log": read_sync_log(settings, slug, limit=100),
            "live_ok": request.query_params.get("live_ok"),
            "live_error": request.query_params.get("live_error"),
            "pnl_currency": str(tenant.config.trader.pnl_currency or "").strip().upper()
            or "USD",
        },
    )


@router.post("/bots/{slug}/trader/live/sync-book")
def trader_live_sync_book(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    """Import open IG positions/orders into the local ledger (manual resync)."""
    from urllib.parse import quote

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    result = adopt_ig_book(session, settings, slug)
    if result.get("ok"):
        return RedirectResponse(
            url=(
                f"/dashboard/bots/{slug}/trader/live/book-sync"
                f"?live_ok={quote(str(result['message']))}"
            ),
            status_code=303,
        )
    return RedirectResponse(
        url=(
            f"/dashboard/bots/{slug}/trader/live/book-sync"
            f"?live_error={quote(str(result['message']))}"
        ),
        status_code=303,
    )


@router.post("/bots/{slug}/trader/live/sync-log/clear")
def trader_live_sync_log_clear(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    from urllib.parse import quote

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    clear_sync_log(settings, slug)
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}/trader/live/book-sync?live_ok={quote('Desync log flushed')}",
        status_code=303,
    )


@router.post("/bots/{slug}/trader/live/clear")
def trader_live_clear(
    slug: str,
    user: User = Depends(require_user),
    tenant_service: TenantService = Depends(get_tenant_service),
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
    session: Session = Depends(get_session),
):
    from urllib.parse import quote

    tenant = _tenant_or_404(tenant_service, slug)
    _require_access(user, user_service, tenant)
    _require_trader_active(tenant, session)
    try:
        clear_live_history(settings, slug)
    except ValueError as exc:
        return RedirectResponse(
            url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_error={quote(str(exc))}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/bots/{slug}?tab=trading&ttab=live&live_ok={quote('Paper history cleared')}",
        status_code=303,
    )


