"""Synchronization primitives for concurrent ML workloads.

Provides thread-safe constructs for coordinating parallel data loading,
model updates, and inference workers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class Lock:
    """Reentrant lock wrapper with timeout support.

    Usage::

        lock = Lock(name="model_update")
        with lock.acquire(timeout=5.0):
            model.update_weights(new_grads)
    """

    def __init__(self, name: str = "", timeout: float = 30.0) -> None:
        self._lock = threading.RLock()
        self.name = name
        self._timeout = timeout
        self._acquired_at: float | None = None

    def acquire(self, timeout: float | None = None) -> _LockContext:
        """Acquire the lock with optional timeout."""
        t = timeout if timeout is not None else self._timeout
        return _LockContext(self._lock, t, self.name)

    @property
    def locked(self) -> bool:
        return self._lock._is_owned()


class _LockContext:
    """Context manager for lock acquisition."""

    def __init__(
        self, lock: threading.RLock, timeout: float, name: str
    ) -> None:
        self._lock = lock
        self._timeout = timeout
        self._name = name
        self.acquired = False

    def __enter__(self) -> _LockContext:
        self.acquired = self._lock.acquire(timeout=self._timeout)
        if not self.acquired:
            raise TimeoutError(
                f"Lock '{self._name}' acquisition timed out after {self._timeout}s"
            )
        return self

    def __exit__(self, *args: Any) -> None:
        if self.acquired:
            self._lock.release()


class Semaphore:
    """Counting semaphore for limiting concurrent access.

    Usage::

        sem = Semaphore(max_concurrent=4)
        for batch in dataloader:
            with sem:
                model.train_step(batch)
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        self._sem = threading.Semaphore(max_concurrent)
        self._max = max_concurrent
        self._count = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        with self._lock:
            return self._max - self._count

    def __enter__(self) -> Semaphore:
        self._sem.acquire()
        with self._lock:
            self._count += 1
        return self

    def __exit__(self, *args: Any) -> None:
        with self._lock:
            self._count -= 1
        self._sem.release()


@dataclass
class _WorkerState:
    """Internal state for a worker."""

    worker_id: int
    active: bool = False
    task_count: int = 0
    error_count: int = 0
    last_error: str = ""


class WorkerPool:
    """Thread-based worker pool for parallel ML tasks.

    Usage::

        pool = WorkerPool(max_workers=4)
        pool.start()

        for batch in batches:
            pool.submit(train_step, batch)

        pool.shutdown()
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._workers: list[threading.Thread] = []
        self._queue: list[tuple[Callable, tuple, dict]] = []
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._states: list[_WorkerState] = []

    def start(self) -> None:
        """Start worker threads."""
        self._stop_event.clear()
        for i in range(self._max_workers):
            state = _WorkerState(worker_id=i)
            self._states.append(state)
            t = threading.Thread(
                target=self._worker_loop, args=(state,), daemon=True
            )
            self._workers.append(t)
            t.start()

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Submit a task to the pool."""
        with self._condition:
            self._queue.append((fn, args, kwargs))
            self._condition.notify()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool."""
        self._stop_event.set()
        if wait:
            for t in self._workers:
                t.join(timeout=5.0)
        self._workers.clear()

    def get_stats(self) -> list[dict[str, Any]]:
        """Get worker statistics."""
        return [
            {
                "worker_id": s.worker_id,
                "active": s.active,
                "task_count": s.task_count,
                "error_count": s.error_count,
            }
            for s in self._states
        ]

    def _worker_loop(self, state: _WorkerState) -> None:
        """Main loop for a worker thread."""
        while not self._stop_event.is_set():
            task = None
            with self._condition:
                while not self._queue and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.1)
                if self._queue:
                    task = self._queue.pop(0)

            if task is None:
                continue

            fn, args, kwargs = task
            state.active = True
            try:
                fn(*args, **kwargs)
                state.task_count += 1
            except Exception as exc:
                state.error_count += 1
                state.last_error = str(exc)
            finally:
                state.active = False
