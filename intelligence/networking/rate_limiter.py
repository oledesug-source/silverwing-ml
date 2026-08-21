"""Rate limiting and request throttling for ML serving endpoints.

Provides token bucket, sliding window, and per-key rate limiters
to protect model serving infrastructure from overload.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_ms: float = 0.0
    key: str = ""


class TokenBucket:
    """Token bucket rate limiter.

    Usage::

        bucket = TokenBucket(rate=100, burst=50)
        if bucket.acquire():
            process_request()
        else:
            return 429
    """

    def __init__(self, rate: float, burst: int = 1) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if allowed."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_acquire(self, tokens: int = 1, timeout: float = 5.0) -> bool:
        """Block until tokens are available or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.acquire(tokens):
                return True
            time.sleep(0.005)
        return False

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class SlidingWindowLimiter:
    """Sliding window rate limiter with per-second and per-minute tracking.

    Usage::

        limiter = SlidingWindowLimiter(per_second=10, per_minute=200)
        result = limiter.check("api-key-123")
        if not result.allowed:
            return Response(status=429, body={"retry_after": result.retry_after_ms})
    """

    def __init__(self, per_second: int = 10, per_minute: int = 200) -> None:
        self._per_second = per_second
        self._per_minute = per_minute
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str = "default") -> RateLimitResult:
        """Check if a request is allowed for the given key."""
        now = time.time()
        with self._lock:
            window = self._windows[key]

            cutoff_minute = now - 60
            while window and window[0] < cutoff_minute:
                window.popleft()

            recent_second = sum(1 for t in window if t > now - 1)
            recent_minute = len(window)

            if recent_second >= self._per_second:
                return RateLimitResult(
                    allowed=False,
                    limit=self._per_second,
                    remaining=0,
                    retry_after_ms=1000.0,
                    key=key,
                )

            if recent_minute >= self._per_minute:
                oldest = window[0] if window else now
                retry = (oldest + 60 - now) * 1000
                return RateLimitResult(
                    allowed=False,
                    limit=self._per_minute,
                    remaining=0,
                    retry_after_ms=max(retry, 100),
                    key=key,
                )

            window.append(now)
            return RateLimitResult(
                allowed=True,
                limit=self._per_second,
                remaining=self._per_second - recent_second - 1,
                key=key,
            )

    def reset(self, key: str | None = None) -> None:
        """Reset counters for a key or all keys."""
        with self._lock:
            if key:
                self._windows.pop(key, None)
            else:
                self._windows.clear()


class PerKeyRateLimiter:
    """Rate limiter that applies different limits per key/client.

    Usage::

        limiter = PerKeyRateLimiter(default_per_second=10)
        limiter.set_limit("premium-user", per_second=100)

        result = limiter.check("premium-user")
    """

    def __init__(self, default_per_second: int = 10) -> None:
        self._default = default_per_second
        self._limits: dict[str, int] = {}
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def set_limit(self, key: str, per_second: int) -> None:
        """Override rate limit for a specific key."""
        with self._lock:
            self._limits[key] = per_second
            self._buckets[key] = TokenBucket(rate=per_second, burst=per_second)

    def check(self, key: str = "default") -> RateLimitResult:
        """Check if a request is allowed."""
        with self._lock:
            if key not in self._buckets:
                rate = self._limits.get(key, self._default)
                self._buckets[key] = TokenBucket(rate=rate, burst=rate)

        bucket = self._buckets[key]
        allowed = bucket.acquire()
        rate = self._limits.get(key, self._default)

        return RateLimitResult(
            allowed=allowed,
            limit=rate,
            remaining=max(0, int(bucket.available)),
            retry_after_ms=0.0 if allowed else (1000.0 / rate),
            key=key,
        )
