from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from evenor.application.tenant_service import TenantService
from evenor.application.remember_me_service import RememberMeService
from evenor.application.user_service import UserService
from evenor.config.settings import Settings, get_settings
from evenor.domain.models.user import User
from evenor.interfaces.api.deps import get_session, get_tenant_service
from evenor.interfaces.web.auth_home import resolve_authenticated_home
from evenor.interfaces.web.deps import get_optional_user, get_remember_me_service, get_user_service
from evenor.interfaces.web.remember_me_cookies import (
    apply_clear_remember_cookie,
    apply_remember_cookie_to_redirect,
)
from evenor.interfaces.web.templates import templates

router = APIRouter(tags=["auth"])


@router.get("/")
def root(
    authenticated: RedirectResponse | None = Depends(resolve_authenticated_home),
) -> RedirectResponse:
    if authenticated is not None:
        return authenticated
    return RedirectResponse(url="/auth/login", status_code=302)


@router.get("/dashboard")
def dashboard_home(
    authenticated: RedirectResponse | None = Depends(resolve_authenticated_home),
) -> RedirectResponse:
    if authenticated is not None:
        return authenticated
    return RedirectResponse(url="/auth/login", status_code=302)


@router.get("/auth/login", response_class=HTMLResponse)
def login_form(
    request: Request,
    error: str | None = None,
    authenticated: RedirectResponse | None = Depends(resolve_authenticated_home),
):
    if authenticated is not None:
        return authenticated
    return templates.TemplateResponse(
        request, "login.html", {"error": error, "title": "Login"}
    )


@router.post("/auth/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    remember_me: str | None = Form(None),
    user_service: UserService = Depends(get_user_service),
    tenant_service: TenantService = Depends(get_tenant_service),
    remember_service: RememberMeService = Depends(get_remember_me_service),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    user = user_service.authenticate(email, password)
    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid credentials", "title": "Login"},
            status_code=401,
        )
    request.session["user_id"] = user.id
    home = user_service.dashboard_home_url(user, tenant_service.list_tenants())
    response = RedirectResponse(url=home, status_code=303)
    if remember_me:
        raw_token = remember_service.issue_token(user.id)
        apply_remember_cookie_to_redirect(
            response,
            remember_service=remember_service,
            settings=settings,
            user_id=user.id,
            raw_token=raw_token,
        )
    else:
        remember_service.revoke_token(user.id)
        apply_clear_remember_cookie(response, settings=settings)
    session.commit()
    return response


@router.post("/auth/logout")
def logout(
    request: Request,
    user: User | None = Depends(get_optional_user),
    remember_service: RememberMeService = Depends(get_remember_me_service),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    if user is not None:
        remember_service.revoke_token(user.id)
    request.session.clear()
    session.commit()
    response = RedirectResponse(url="/auth/login", status_code=303)
    return apply_clear_remember_cookie(response, settings=settings)
