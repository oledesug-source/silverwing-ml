"""
Middleware framework for request/response processing pipeline.
"""

from __future__ import annotations

import time
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from threading import Lock
from typing import Any

__all__ = [
    "Middleware",
    "CORSMiddleware",
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    "LoggingMiddleware",
    "SecurityHeadersMiddleware",
    "CompressionMiddleware",
    "SessionMiddleware",
    "MiddlewarePipeline",
]


class Middleware(ABC):
    """Base class for request/response middleware."""

    @abstractmethod
    def process_request(self, request: Any) -> Any:
        """Process and optionally modify the incoming request."""

    @abstractmethod
    def process_response(self, response: Any) -> Any:
        """Process and optionally modify the outgoing response."""


class CORSMiddleware(Middleware):
    """Cross-Origin Resource Sharing middleware."""

    def __init__(
        self,
        allow_origins: list[str] | None = None,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
        expose_headers: list[str] | None = None,
        max_age: int = 86400,
        allow_credentials: bool = False,
    ) -> None:
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.allow_headers = allow_headers or ["Content-Type", "Authorization", "X-Requested-With"]
        self.expose_headers = expose_headers or []
        self.max_age = max_age
        self.allow_credentials = allow_credentials

    def process_request(self, request: Any) -> Any:
        return request

    def process_response(self, response: Any) -> Any:
        origin = getattr(_request := response, "_cors_origin", None)
        if not hasattr(response, "headers"):
            return response
        origin_header = self._determine_origin(origin)
        if origin_header:
            response.headers["Access-Control-Allow-Origin"] = origin_header
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            if self.expose_headers:
                response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
            response.headers["Access-Control-Max-Age"] = str(self.max_age)
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    def _determine_origin(self, origin: str | None) -> str | None:
        if "*" in self.allow_origins:
            return "*"
        if origin and origin in self.allow_origins:
            return origin
        return None


class AuthenticationMiddleware(Middleware):
    """Token-based authentication middleware with public path exemptions."""

    def __init__(self, validate_token: Callable[[str], Any | None], public_paths: list[str] | None = None) -> None:
        self.validate_token = validate_token
        self.public_paths = public_paths or []

    def process_request(self, request: Any) -> Any:
        path = getattr(request, "path", "/")
        if any(path.startswith(p) for p in self.public_paths):
            request.user = None
            return request
        auth_header = request.headers.get("Authorization", "") if hasattr(request, "headers") else ""
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        elif auth_header.startswith("Token "):
            token = auth_header[6:]
        user = self.validate_token(token) if token else None
        request.user = user
        request.auth_error = None if user else ("Authentication required" if token else "No token provided")
        return request

    def process_response(self, response: Any) -> Any:
        return response


class RateLimitMiddleware(Middleware):
    """Rate limiting middleware tracking requests per IP address."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def process_request(self, request: Any) -> Any:
        ip = getattr(request, "remote_addr", "127.0.0.1")
        now = time.time()
        with self._lock:
            self._requests[ip] = [t for t in self._requests[ip] if now - t < self.window_seconds]
            if len(self._requests[ip]) >= self.max_requests:
                request._rate_limited = True
                request._retry_after = int(self._requests[ip][0] + self.window_seconds - now) + 1
            else:
                self._requests[ip].append(now)
                request._rate_limited = False
        return request

    def process_response(self, response: Any) -> Any:
        if hasattr(response, "headers") and hasattr(response, "status_code"):
            if getattr(response, "_rate_limited", False):
                response.headers["Retry-After"] = str(getattr(response, "_retry_after", self.window_seconds))
        return response


class LoggingMiddleware(Middleware):
    """Request/response logging middleware."""

    def __init__(self, log_fn: Callable[[str], None] | None = None, fmt: str = "{method} {path} {status}") -> None:
        self.log_fn = log_fn or print
        self.fmt = fmt

    def process_request(self, request: Any) -> Any:
        request._start_time = time.time()
        return request

    def process_response(self, response: Any) -> Any:
        elapsed = 0.0
        if hasattr(response, "_start_time"):
            elapsed = time.time() - response._start_time
        msg = self.fmt.format(
            method=getattr(response, "_method", "GET"),
            path=getattr(response, "_path", "/"),
            status=getattr(response, "status_code", 200),
            elapsed=f"{elapsed:.3f}s",
        )
        self.log_fn(msg)
        return response


class SecurityHeadersMiddleware(Middleware):
    """Security headers middleware adding standard protective headers."""

    def __init__(
        self,
        content_type_options: str = "nosniff",
        frame_options: str = "DENY",
        xss_protection: str = "1; mode=block",
        content_security_policy: str = "default-src 'self'",
        strict_transport: bool = True,
        referrer_policy: str = "strict-origin-when-cross-origin",
    ) -> None:
        self.headers = {
            "X-Content-Type-Options": content_type_options,
            "X-Frame-Options": frame_options,
            "X-XSS-Protection": xss_protection,
            "Content-Security-Policy": content_security_policy,
            "Referrer-Policy": referrer_policy,
        }
        if strict_transport:
            self.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    def process_request(self, request: Any) -> Any:
        return request

    def process_response(self, response: Any) -> Any:
        if hasattr(response, "headers"):
            for key, value in self.headers.items():
                response.headers[key] = value
        return response


class CompressionMiddleware(Middleware):
    """Response body compression middleware using zlib deflate."""

    def __init__(self, threshold: int = 500, level: int = 6) -> None:
        self.threshold = threshold
        self.level = level

    def process_request(self, request: Any) -> Any:
        if hasattr(request, "headers"):
            accept = request.headers.get("Accept-Encoding", "")
            request._accepts_gzip = "gzip" in accept or "deflate" in accept
        else:
            request._accepts_gzip = False
        return request

    def process_response(self, response: Any) -> Any:
        if not hasattr(response, "body") or not hasattr(response, "headers"):
            return response
        body = response.body
        if len(body) < self.threshold:
            return response
        compressed = zlib.compress(body, self.level)
        if len(compressed) < len(body):
            response.body = compressed
            response.headers["Content-Encoding"] = "deflate"
            response.headers["Content-Length"] = str(len(compressed))
        return response


class SessionMiddleware(Middleware):
    """Cookie-based session middleware with in-memory storage."""

    def __init__(self, session_cookie: str = "session_id", secret_key: str = "session-secret") -> None:
        self.session_cookie = session_cookie
        self.secret_key = secret_key
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def _generate_session_id(self) -> str:
        import hashlib
        import uuid
        return hashlib.sha256(uuid.uuid4().bytes + self.secret_key.encode()).hexdigest()[:32]

    def process_request(self, request: Any) -> Any:
        cookies = request.cookies() if callable(getattr(request, "cookies", None)) else {}
        session_id = cookies.get(self.session_cookie, "")
        with self._lock:
            if session_id and session_id in self._sessions:
                request.session = self._sessions[session_id]
                request._session_id = session_id
            else:
                new_id = self._generate_session_id()
                self._sessions[new_id] = {}
                request.session = self._sessions[new_id]
                request._session_id = new_id
        return request

    def process_response(self, response: Any) -> Any:
        if hasattr(response, "set_cookie") and hasattr(response, "_session_id"):
            response.set_cookie(self.session_cookie, response._session_id, httponly=True)
        return response


class MiddlewarePipeline:
    """Ordered pipeline that chains middleware and a final handler."""

    def __init__(self) -> None:
        self._middlewares: list[Middleware] = []

    def use(self, middleware: Middleware) -> MiddlewarePipeline:
        """Add a middleware to the pipeline."""
        self._middlewares.append(middleware)
        return self

    def execute(self, request: Any, handler: Callable[..., Any]) -> Any:
        """Execute the pipeline: run all request middleware, call handler, then run response middleware."""
        for mw in self._middlewares:
            request = mw.process_request(request)
        response = handler(request)
        for mw in reversed(self._middlewares):
            response = mw.process_response(response)
        return response
