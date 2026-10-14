"""Request-id + access-log middleware.

Written as plain ASGI middleware rather than Starlette's `BaseHTTPMiddleware`, which buffers and
interferes with streaming responses and client disconnects (we rely on both for SSE chat).
"""

import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import error_body
from app.core.logging import request_id_var
from app.core.responses import UTF8JSONResponse as JSONResponse

logger = logging.getLogger("app.request")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Reuse an incoming id (e.g. from a proxy) only if it is short and safe to log.
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        request_id_var.set(request_id)

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            path = scope.get("path", "")
            # Health probes run every few seconds on hosting platforms; keep them out of INFO.
            level = logging.DEBUG if path.startswith("/api/health") else logging.INFO
            logger.log(
                level,
                "%s %s -> %s",
                scope.get("method"),
                path,
                status_code,
                extra={"duration_ms": duration_ms},
            )


class UploadSizeLimitMiddleware:
    """Reject oversized uploads from the Content-Length header, before the body is read.

    Without this, the multipart parser would spool a multi-GB request to disk before our
    endpoint could reject it. Requests without Content-Length (chunked) are still caught by the
    endpoint's own streaming size check.
    """

    # Multipart framing (boundaries, part headers) adds a little on top of the file itself.
    MULTIPART_OVERHEAD_BYTES = 64 * 1024

    def __init__(self, app: ASGIApp, *, max_bytes: int, path: str = "/api/documents") -> None:
        self.app = app
        self.max_bytes = max_bytes + self.MULTIPART_OVERHEAD_BYTES
        self.limit_mb = max_bytes // (1024 * 1024)
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == self.path:
            raw = dict(scope["headers"]).get(b"content-length")
            if raw is not None and raw.isdigit() and int(raw) > self.max_bytes:
                response = JSONResponse(
                    error_body(
                        "FILE_TOO_LARGE", f"The file is larger than the {self.limit_mb} MB limit."
                    ),
                    status_code=413,
                    headers={"Connection": "close"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
