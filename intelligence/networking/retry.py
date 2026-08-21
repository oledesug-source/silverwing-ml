"""Retry logic with exponential backoff for resilient API calls.

Handles transient failures in ML serving pipelines with configurable
retry strategies, jitter, and circuit breaking.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 10000.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)


@dataclass
class RetryResult:
    """Result of a retry attempt."""

    success: bool
    value: Any = None
    attempts: int = 0
    total_delay_ms: float = 0.0
    last_error: str = ""


class RetryExecutor:
    """Executes functions with automatic retry and backoff.

    Usage::

        executor = RetryExecutor(RetryConfig(max_retries=3))

        def call_api():
            return requests.get("http://model-server/predict")

        result = executor.execute(call_api)
        if result.success:
            print(result.value)
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self._config = config or RetryConfig()

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute fn with retries."""
        config = self._config
        last_error: BaseException | None = None
        total_delay = 0.0

        for attempt in range(config.max_retries + 1):
            try:
                value = fn(*args, **kwargs)
                return RetryResult(
                    success=True,
                    value=value,
                    attempts=attempt + 1,
                    total_delay_ms=total_delay,
                )
            except config.retryable_exceptions as exc:
                last_error = exc
                if attempt < config.max_retries:
                    delay = self._compute_delay(attempt)
                    total_delay += delay
                    time.sleep(delay / 1000.0)

        return RetryResult(
            success=False,
            attempts=config.max_retries + 1,
            total_delay_ms=total_delay,
            last_error=str(last_error) if last_error else "",
        )

    def _compute_delay(self, attempt: int) -> float:
        config = self._config
        delay = config.base_delay_ms * (config.exponential_base ** attempt)
        delay = min(delay, config.max_delay_ms)
        if config.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    Usage::

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        if breaker.allow_request():
            try:
                result = call_external_service()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            # Circuit is open, use fallback
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_attempts = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                return self._half_open_attempts < self._half_open_max
            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }
