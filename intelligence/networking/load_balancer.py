"""Load balancer for distributing requests across ML inference backends.

Provides round-robin and least-connections strategies with health
checking for distributing inference load.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Backend:
    """An inference backend endpoint."""

    url: str
    name: str = ""
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_health_check: float = 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests


class LoadBalancer:
    """Distributes requests across backends using various strategies.

    Usage::

        lb = LoadBalancer(strategy="round_robin")
        lb.add_backend(Backend(url="http://gpu1:8000", name="gpu1"))
        lb.add_backend(Backend(url="http://gpu2:8000", name="gpu2"))

        backend = lb.next_backend()
        # forward request to backend.url
        lb.record_success(backend)
    """

    STRATEGIES = ("round_robin", "least_connections", "weighted")

    def __init__(
        self,
        strategy: str = "round_robin",
        health_check_interval: float = 30.0,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self._strategy = strategy
        self._health_check_interval = health_check_interval
        self._backends: list[Backend] = []
        self._index = 0
        self._lock = threading.Lock()

    def add_backend(self, backend: Backend) -> None:
        """Register a backend."""
        with self._lock:
            self._backends.append(backend)

    def remove_backend(self, url: str) -> None:
        """Remove a backend by URL."""
        with self._lock:
            self._backends = [b for b in self._backends if b.url != url]

    def next_backend(self) -> Backend:
        """Select the next backend based on strategy."""
        with self._lock:
            healthy = [b for b in self._backends if b.healthy]
            if not healthy:
                raise RuntimeError("No healthy backends available")

            if self._strategy == "round_robin":
                backend = healthy[self._index % len(healthy)]
                self._index += 1
            elif self._strategy == "least_connections":
                backend = min(healthy, key=lambda b: b.active_connections)
            elif self._strategy == "weighted":
                total_weight = sum(b.weight for b in healthy)
                if total_weight == 0:
                    backend = healthy[0]
                else:
                    import random
                    r = random.random() * total_weight
                    cumulative = 0
                    backend = healthy[0]
                    for b in healthy:
                        cumulative += b.weight
                        if r <= cumulative:
                            backend = b
                            break
            else:
                backend = healthy[0]

            backend.active_connections += 1
            return backend

    def record_success(self, backend: Backend) -> None:
        """Record a successful request."""
        with self._lock:
            backend.active_connections = max(0, backend.active_connections - 1)
            backend.total_requests += 1

    def record_error(self, backend: Backend) -> None:
        """Record a failed request."""
        with self._lock:
            backend.active_connections = max(0, backend.active_connections - 1)
            backend.total_requests += 1
            backend.total_errors += 1

    def health_check(self) -> dict[str, bool]:
        """Check health of all backends. Returns {url: healthy}."""
        results: dict[str, bool] = {}
        now = time.time()
        with self._lock:
            for backend in self._backends:
                if now - backend.last_health_check < self._health_check_interval:
                    results[backend.url] = backend.healthy
                    continue
                backend.last_health_check = now
                backend.healthy = backend.error_rate < 0.5
                results[backend.url] = backend.healthy
        return results

    def get_stats(self) -> list[dict[str, Any]]:
        """Get stats for all backends."""
        with self._lock:
            return [
                {
                    "name": b.name,
                    "url": b.url,
                    "healthy": b.healthy,
                    "active": b.active_connections,
                    "total": b.total_requests,
                    "errors": b.total_errors,
                    "error_rate": f"{b.error_rate:.2%}",
                }
                for b in self._backends
            ]
