from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

def _http_status_from_gemini_code(code: Any) -> int:
    if isinstance(code, int) and 400 <= code <= 599:
        return code
    return 502


def register_exception_handlers(app: FastAPI) -> None:
    """Expose Google GenAI API failures as JSON with HTTP status (e.g. 429) instead of a generic 500."""

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
