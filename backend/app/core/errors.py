"""Error model and exception handlers.

Every error response, expected or not, has the same JSON shape:

    {"error": {"code": "DOCUMENT_NOT_FOUND", "message": "...", "details": {}, "request_id": "..."}}

Services raise `AppError` (or a subclass) with a stable machine-readable `code`; the frontend
switches on the code and shows the human-readable message.
"""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_var
from app.core.responses import UTF8JSONResponse as JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """An expected, user-facing error."""

    status_code: int = 400
    code: str = "BAD_REQUEST"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class ConflictError(AppError):
    status_code = 409
    code = "CONFLICT"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id_var.get(),
        }
    }


async def _app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    logger.info("app error", extra={"code": exc.code, "status": exc.status_code})
    return JSONResponse(error_body(exc.code, exc.message, exc.details), status_code=exc.status_code)


async def _http_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    try:
        code = HTTPStatus(exc.status_code).name
    except ValueError:
        code = "HTTP_ERROR"
    message = exc.detail if isinstance(exc.detail, str) else code.replace("_", " ").capitalize()
    return JSONResponse(error_body(code, message), status_code=exc.status_code, headers=exc.headers)


async def _validation_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(
        error_body("VALIDATION_ERROR", "The request was invalid.", {"errors": errors}),
        status_code=422,
    )


async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    # Full traceback goes to the logs; the client gets a generic message (no internals leaked).
    logger.exception("unhandled error", exc_info=exc)
    return JSONResponse(
        error_body("INTERNAL_ERROR", "Something went wrong on our side. Please try again."),
        status_code=500,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
