from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from chatbot.application.user_service import UserService
from chatbot.interfaces.api.deps import get_session
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
    return RedirectResponse(url="/dashboard/bots", status_code=303)


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/auth/login", status_code=303)
