"""HTTP middleware pipeline for request/response processing.

Provides composable middleware for logging, authentication, CORS,
compression, and custom request transformation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MiddlewareContext:
    """Context passed through the middleware pipeline."""

    method: str = ""
    path: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""
    query: dict[str, str] = field(default_factory=dict)
    status: int = 200
    response_body: Any = None
    response_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stopped: bool = False


MiddlewareFn = Callable[[MiddlewareContext], MiddlewareContext | None]


class MiddlewarePipeline:
    """Composable middleware pipeline for HTTP requests.

    Usage::

        pipeline = MiddlewarePipeline()
        pipeline.add(LoggingMiddleware())
        pipeline.add(CorsMiddleware(allow_origins=["*"]))
        pipeline.add(AuthMiddleware(verify_token=my_verify))

        ctx = pipeline.process(MiddlewareContext(
            method="POST", path="/predict", body='{"x": 1}'
        ))
    """

    def __init__(self) -> None:
        self._before: list[MiddlewareFn] = []
        self._after: list[MiddlewareFn] = []

    def add(
        self,
        middleware: MiddlewareFn | BaseMiddleware,
    ) -> MiddlewarePipeline:
        """Add a middleware. Accepts a function or a BaseMiddleware instance."""
        if isinstance(middleware, BaseMiddleware):
            if middleware.before:
                self._before.append(middleware.before)
            if middleware.after:
                self._after.append(middleware.after)
        else:
            self._before.append(middleware)
        return self

    def process(self, ctx: MiddlewareContext) -> MiddlewareContext:
        """Run the full middleware pipeline."""
        for fn in self._before:
            result = fn(ctx)
            if result is not None:
                ctx = result
            if ctx.stopped:
                return ctx

        for fn in reversed(self._after):
            result = fn(ctx)
            if result is not None:
                ctx = result

        return ctx


class BaseMiddleware:
    """Base class for middleware with before/after hooks."""

    @property
    def before(self) -> MiddlewareFn | None:
        return None

    @property
    def after(self) -> MiddlewareFn | None:
        return None


class LoggingMiddleware(BaseMiddleware):
    """Logs request timing and status."""

    def __init__(self) -> None:
        self._start: float = 0

    @property
    def before(self) -> MiddlewareFn:
        def _before(ctx: MiddlewareContext) -> MiddlewareContext:
            self._start = time.time()
            return ctx
        return _before

    @property
    def after(self) -> MiddlewareFn:
        def _after(ctx: MiddlewareContext) -> MiddlewareContext:
            elapsed_ms = (time.time() - self._start) * 1000
            ctx.metadata["elapsed_ms"] = elapsed_ms
            ctx.metadata["log"] = (
                f"{ctx.method} {ctx.path} -> {ctx.status} ({elapsed_ms:.1f}ms)"
            )
            return ctx
        return _after


class CorsMiddleware(BaseMiddleware):
    """Adds CORS headers to responses."""

    def __init__(
        self,
        allow_origins: list[str] | None = None,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
        max_age: int = 86400,
    ) -> None:
        self._origins = allow_origins or ["*"]
        self._methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self._headers = allow_headers or ["Content-Type", "Authorization"]
        self._max_age = max_age

    @property
    def before(self) -> MiddlewareFn:
        def _before(ctx: MiddlewareContext) -> MiddlewareContext:
            origin = ctx.headers.get("Origin", ctx.headers.get("origin", "*"))
            if "*" in self._origins or origin in self._origins:
                ctx.response_headers["Access-Control-Allow-Origin"] = origin
                ctx.response_headers["Access-Control-Allow-Methods"] = ", ".join(self._methods)
                ctx.response_headers["Access-Control-Allow-Headers"] = ", ".join(self._headers)
                ctx.response_headers["Access-Control-Max-Age"] = str(self._max_age)
            if ctx.method == "OPTIONS":
                ctx.status = 204
                ctx.stopped = True
            return ctx
        return _before


class AuthMiddleware(BaseMiddleware):
    """Validates authentication tokens/keys."""

    def __init__(
        self,
        verify_token: Callable[[str], bool],
        header: str = "Authorization",
        skip_paths: list[str] | None = None,
    ) -> None:
        self._verify = verify_token
        self._header = header
        self._skip = set(skip_paths or ["/health", "/info"])

    @property
    def before(self) -> MiddlewareFn:
        def _before(ctx: MiddlewareContext) -> MiddlewareContext:
            if ctx.path in self._skip:
                return ctx
            token = ctx.headers.get(self._header, "")
            token = token.removeprefix("Bearer ").strip()
            if not self._verify(token):
                ctx.status = 401
                ctx.response_body = {"error": "Unauthorized"}
                ctx.stopped = True
            return ctx
        return _before


class RequestIdMiddleware(BaseMiddleware):
    """Adds a unique request ID to each request."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = __import__("threading").Lock()

    @property
    def before(self) -> MiddlewareFn:
        def _before(ctx: MiddlewareContext) -> MiddlewareContext:
            with self._lock:
                self._counter += 1
            ctx.metadata["request_id"] = f"req-{self._counter:08d}"
            ctx.response_headers["X-Request-Id"] = ctx.metadata["request_id"]
            return ctx
        return _before
