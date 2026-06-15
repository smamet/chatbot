from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from chatbot.application.tenant_service import TenantService
from chatbot.application.user_service import UserService
from chatbot.interfaces.api.deps import get_session, get_tenant_service
from chatbot.interfaces.web.deps import get_user_service
from chatbot.interfaces.web.templates import templates

router = APIRouter(tags=["auth"])


@router.get("/auth/login", response_class=HTMLResponse)
def login_form(request: Request, error: str | None = None):
    return templates.TemplateResponse(
        request, "login.html", {"error": error, "title": "Login"}
    )


@router.post("/auth/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    user_service: UserService = Depends(get_user_service),
    tenant_service: TenantService = Depends(get_tenant_service),
    session: Session = Depends(get_session),
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
    session.commit()
    home = user_service.dashboard_home_url(user, tenant_service.list_tenants())
    return RedirectResponse(url=home, status_code=303)


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/auth/login", status_code=303)
