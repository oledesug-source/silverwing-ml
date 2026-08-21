"""Connection pooling for HTTP clients and database connections.

Provides reusable connection management to reduce overhead
in high-throughput ML serving scenarios.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PooledConnection:
    """A wrapper around a pooled connection."""

    connection: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    in_use: bool = False

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used


class ConnectionPool:
    """Generic connection pool with lifecycle management.

    Usage::

        pool = ConnectionPool(
            factory=lambda: http.client.HTTPConnection("localhost"),
            max_size=10,
            max_age_seconds=300,
        )

        with pool.connection() as conn:
            conn.request("GET", "/predict")
            resp = conn.getresponse()
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        max_age_seconds: float = 300.0,
        validate: Callable[[Any], bool] | None = None,
    ) -> None:
        self._factory = factory
        self._max_size = max_size
        self._max_age = max_age_seconds
        self._validate = validate
        self._pool: queue.Queue[PooledConnection] = queue.Queue(maxsize=max_size)
        self._all_connections: list[PooledConnection] = []
        self._lock = threading.Lock()
        self._stats = {"created": 0, "reused": 0, "expired": 0, "failed": 0}

    def acquire(self) -> PooledConnection:
        """Acquire a connection from the pool."""
        while True:
            try:
                pooled = self._pool.get_nowait()
            except queue.Empty:
                break

            if self._is_valid(pooled):
                pooled.in_use = True
                pooled.last_used = time.time()
                pooled.use_count += 1
                self._stats["reused"] += 1
                return pooled
            else:
                self._stats["expired"] += 1
                self._remove(pooled)

        with self._lock:
            if len(self._all_connections) < self._max_size:
                conn = self._factory()
                pooled = PooledConnection(connection=conn)
                self._all_connections.append(pooled)
                self._stats["created"] += 1
                pooled.in_use = True
                return pooled

        pooled = self._pool.get(timeout=5.0)
        if self._is_valid(pooled):
            pooled.in_use = True
            pooled.last_used = time.time()
            pooled.use_count += 1
            return pooled

        self._stats["failed"] += 1
        raise ConnectionError("No valid connections available")

    def release(self, pooled: PooledConnection) -> None:
        """Release a connection back to the pool."""
        pooled.in_use = False
        pooled.last_used = time.time()
        if self._is_valid(pooled):
            try:
                self._pool.put_nowait(pooled)
            except queue.Full:
                self._remove(pooled)
        else:
            self._remove(pooled)

    def _is_valid(self, pooled: PooledConnection) -> bool:
        if pooled.age_seconds > self._max_age:
            return False
        if self._validate and not self._validate(pooled.connection):
            return False
        return True

    def _remove(self, pooled: PooledConnection) -> None:
        with self._lock:
            if pooled in self._all_connections:
                self._all_connections.remove(pooled)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                self._remove(pooled)
            except queue.Empty:
                break
        with self._lock:
            self._all_connections.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Pool statistics."""
        available = self._pool.qsize()
        with self._lock:
            total = len(self._all_connections)
        in_use = total - available
        return {**self._stats, "total": total, "available": available, "in_use": in_use}
