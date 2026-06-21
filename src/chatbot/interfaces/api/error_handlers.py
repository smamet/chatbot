from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from chatbot.config.settings import get_settings
from chatbot.interfaces.web.templates import templates

logger = logging.getLogger(__name__)

_STATUS_TITLES = {
    400: "Bad request",
    401: "Not authenticated",
    403: "Access denied",
    404: "Page not found",
    422: "Invalid input",
    500: "Server error",
}


def _http_status_from_gemini_code(code: Any) -> int:
    if isinstance(code, int) and 400 <= code <= 599:
        return code
    return 502


def _wants_html_response(request: Request) -> bool:
    path = request.url.path
    if path.startswith("/dashboard") or path.startswith("/auth"):
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept.split(",")[0]


def _status_title(status_code: int) -> str:
    return _STATUS_TITLES.get(status_code, "Error")


def _detail_message(detail: Any) -> str:
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            return str(message)
        return str(detail)
    if isinstance(detail, list):
        return str(detail)
    return str(detail)


def _html_error_response(
    request: Request,
    *,
    status_code: int,
    summary: str,
    exc: BaseException | None = None,
    show_traceback: bool = False,
) -> HTMLResponse:
    traceback_text: str | None = None
    exc_type: str | None = None
    exc_message: str | None = None
    if exc is not None:
        exc_type = type(exc).__name__
        exc_message = str(exc) or None
        if show_traceback:
            traceback_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
    return templates.TemplateResponse(
        request,
        "errors/error.html",
        {
            "title": f"{status_code} — {_status_title(status_code)}",
            "status_code": status_code,
            "status_title": _status_title(status_code),
            "summary": summary,
            "dev_mode": show_traceback,
            "exc_type": exc_type,
            "exc_message": exc_message,
            "traceback": traceback_text,
            "user": None,
        },
        status_code=status_code,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Expose API failures as JSON; dashboard/auth routes get HTML error pages."""

    from google.genai.errors import APIError as GenaiAPIError

    @app.exception_handler(GenaiAPIError)
    async def _handle_genai_api_error(_request: Request, exc: GenaiAPIError) -> JSONResponse:
        status = _http_status_from_gemini_code(exc.code)
        message = (exc.message or str(exc)).strip()
        detail: dict[str, Any] = {
            "kind": "gemini_api",
            "code": exc.code,
            "status": exc.status,
            "message": message,
        }
        return JSONResponse(status_code=status, content={"detail": detail})

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException) -> HTMLResponse | JSONResponse:
        if not _wants_html_response(request):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        settings = get_settings()
        summary = _detail_message(exc.detail) or _status_title(exc.status_code)
        show_traceback = settings.dev_mode and exc.status_code >= 500
        return _html_error_response(
            request,
            status_code=exc.status_code,
            summary=summary,
            exc=exc if show_traceback else None,
            show_traceback=show_traceback,
        )

    @app.exception_handler(Exception)
    async def _handle_unhandled_exception(request: Request, exc: Exception) -> HTMLResponse | JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        settings = get_settings()
        if not _wants_html_response(request):
            if settings.dev_mode:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": {
                            "kind": "internal",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    },
                )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})
        summary = (
            str(exc)
            if settings.dev_mode
            else "Something went wrong. Try again or contact support."
        )
        return _html_error_response(
            request,
            status_code=500,
            summary=summary,
            exc=exc,
            show_traceback=settings.dev_mode,
        )
