"""ASGI Middleware: request-ID tracing, timing, and global error handling."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger("support_agent.middleware")

REQUEST_ID_HEADER = "X-Request-ID"
# Context var so loggers / code anywhere in the call stack can read it.
from contextvars import ContextVar  # noqa: E402

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Return the current request's trace ID (None outside a request)."""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns an X-Request-ID to every request for end-to-end tracing.

    - Honours an incoming ``X-Request-ID`` header (from a gateway) or
      generates a fresh UUID.
    - Echoes the ID back on the response.
    - Logs method / path / status / duration per request.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request_id_ctx.set(request_id)
        request.state.request_id = request_id

        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "UNHANDLED | request_id=%s method=%s path=%s",
                request_id, request.method, request.url.path,
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error_type": "InternalServerError",
                    "message": "An unexpected server error occurred. Please try again later.",
                    "request_id": request_id,
                    "path": request.url.path,
                },
                headers={REQUEST_ID_HEADER: request_id},
            )

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "%s %s -> %d | %sms | request_id=%s",
            request.method, request.url.path, response.status_code,
            elapsed_ms, request_id,
        )
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch-all error handler producing uniform JSON error envelopes.

    HTTPException / RequestValidationError are handled natively by FastAPI;
    this covers domain errors and unexpected exceptions that escape routers,
    guaranteeing every error response carries a ``request_id`` for tracing.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        try:
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001
            request_id = get_request_id() or uuid.uuid4().hex[:16]
            logger.exception(
                "ERROR HANDLED | request_id=%s path=%s exc=%s",
                request_id, request.url.path, exc,
            )
            detail = getattr(exc, "message", None) or getattr(exc, "detail", None)
            return JSONResponse(
                status_code=getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
                or status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error_type": exc.__class__.__name__,
                    "message": detail or "An unexpected server error occurred.",
                    "request_id": request_id,
                    "path": request.url.path,
                },
                headers={REQUEST_ID_HEADER: request_id},
            )


def register_middlewares(app: FastAPI) -> None:
    """Attach tracing + error middleware. Call before mounting routers."""
    # NOTE: with BaseHTTPMiddleware the LAST added middleware runs FIRST.
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestIDMiddleware)


__all__ = [
    "ErrorHandlingMiddleware",
    "RequestIDMiddleware",
    "get_request_id",
    "register_middlewares",
    "REQUEST_ID_HEADER",
]
