"""Networking fundamentals for ML serving and distributed training.

Covers:

- **HTTP** — request/response, status codes, headers
- **REST APIs** — design patterns, versioning, error handling
- **Load balancing** — round-robin, least-connections, weighted, health checks
- **Rate limiting** — token bucket, sliding window, per-key limits
- **Retry/backoff** — exponential backoff, jitter, circuit breaker
- **Connection pooling** — reuse, lifecycle, validation
- **Middleware** — logging, CORS, auth, request IDs
"""

from .connection_pool import ConnectionPool, PooledConnection
from .http_client import HttpClient, HttpResponse
from .load_balancer import Backend, LoadBalancer
from .middleware import (
    AuthMiddleware,
    CorsMiddleware,
    LoggingMiddleware,
    MiddlewareContext,
    MiddlewarePipeline,
    RequestIdMiddleware,
)
from .rate_limiter import PerKeyRateLimiter, RateLimitResult, SlidingWindowLimiter, TokenBucket
from .retry import CircuitBreaker, CircuitState, RetryConfig, RetryExecutor, RetryResult
from .server import Request, Response, Server

__all__ = [
    "HttpClient",
    "HttpResponse",
    "Server",
    "Request",
    "Response",
    "LoadBalancer",
    "Backend",
    "TokenBucket",
    "SlidingWindowLimiter",
    "PerKeyRateLimiter",
    "RateLimitResult",
    "RetryExecutor",
    "RetryConfig",
    "RetryResult",
    "CircuitBreaker",
    "CircuitState",
    "ConnectionPool",
    "PooledConnection",
    "MiddlewarePipeline",
    "MiddlewareContext",
    "LoggingMiddleware",
    "CorsMiddleware",
    "AuthMiddleware",
    "RequestIdMiddleware",
]
