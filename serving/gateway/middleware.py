"""Gateway middleware — rate limiting and structured request logging.

Fills the ``serving.gateway`` package: install onto any FastAPI app::

    from serving.gateway.middleware import install_gateway

    app = create_app(...)
    install_gateway(app)

Tuning via environment:
    SILVERWING_RATE_LIMIT_RPM     requests per minute per client IP (default 120)
    SILVERWING_GATEWAY_LOGGING    "1" enables request logs (default "1")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("serving.gateway")

# ---------------------------------------------------------------------------
# Rate limiter — token bucket per client IP
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe sliding-window counter keyed by client IP."""

    def __init__(self, requests_per_minute: int = 120) -> None:
        self._rpm = max(1, requests_per_minute)
        self._window_seconds = 60.0
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                self._hits[key] = [now]
                # opportunistic eviction keeps the map bounded
                if len(self._hits) > 10_000:
                    self._hits = {
                        k: v for k, v in list(self._hits.items())[-5_000:]
                    }
                return True
            while hits and hits[0] < cutoff:
                hits.pop(0)
            if len(hits) >= self._rpm:
                return False
            hits.append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rejects clients exceeding the per-IP request budget with HTTP 429."""

    def __init__(self, app: ASGIApp, limiter: RateLimiter) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        key = self._limiter.client_ip(request)
        if not self._limiter.allow(key):
            return JSONResponse(
                {"success": False, "error": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "30"},
            )
        return await call_next(request)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status, latency for every HTTP request."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        t0 = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
        )
        return response


def install_gateway(
    app: FastAPI,
    requests_per_minute: int | None = None,
    enable_logging: bool | None = None,
) -> None:
    """Attach gateway middleware to *app* (call after routes are defined)."""
    rpm = requests_per_minute or int(
        os.environ.get("SILVERWING_RATE_LIMIT_RPM", "120")
    )
    do_log = enable_logging
    if do_log is None:
        do_log = os.environ.get("SILVERWING_GATEWAY_LOGGING", "1") == "1"
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(rpm))
    if do_log:
        app.add_middleware(RequestLogMiddleware)


__all__ = [
    "RateLimiter",
    "RateLimitMiddleware",
    "RequestLogMiddleware",
    "install_gateway",
]
