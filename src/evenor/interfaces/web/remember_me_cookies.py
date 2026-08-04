from __future__ import annotations

from fastapi import Response
from fastapi.responses import RedirectResponse

from evenor.application.remember_me_service import REMEMBER_COOKIE_NAME, RememberMeService
from evenor.config.settings import Settings


def attach_remember_cookie(
    response: Response,
    *,
    remember_service: RememberMeService,
    settings: Settings,
    user_id: int,
    raw_token: str,
) -> None:
    signed = remember_service.sign_cookie(user_id, raw_token)
    max_age = remember_service.cookie_max_age_seconds()
    response.set_cookie(
        key=REMEMBER_COOKIE_NAME,
        value=signed,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=not settings.dev_mode,
        path="/",
    )


def clear_remember_cookie(response: Response, *, settings: Settings) -> None:
    response.delete_cookie(
        key=REMEMBER_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=not settings.dev_mode,
    )


def apply_remember_cookie_to_redirect(
    response: RedirectResponse,
    *,
    remember_service: RememberMeService,
    settings: Settings,
    user_id: int,
    raw_token: str,
) -> RedirectResponse:
    attach_remember_cookie(
        response,
        remember_service=remember_service,
        settings=settings,
        user_id=user_id,
        raw_token=raw_token,
    )
    return response


def apply_clear_remember_cookie(
    response: RedirectResponse,
    *,
    settings: Settings,
) -> RedirectResponse:
    clear_remember_cookie(response, settings=settings)
    return response
