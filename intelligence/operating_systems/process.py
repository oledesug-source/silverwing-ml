"""Process lifecycle management for ML workloads.

Provides a lightweight abstraction over Python multiprocessing and
threading for managing concurrent training jobs, data pipelines, and
inference workers.
"""

from __future__ import annotations

import enum
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ProcessState(enum.Enum):
    """Lifecycle states for a managed process."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ProcessInfo:
    """Metadata about a managed process."""

    pid: int | None = None
    name: str = ""
    state: ProcessState = ProcessState.IDLE
    started_at: float | None = None
    exit_code: int | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessManager:
    """Manages a pool of worker processes for parallel ML tasks.

    Usage::

        pm = ProcessManager(max_workers=4)
        pm.start()
        pm.submit(my_training_fn, arg1, arg2)
        pm.shutdown()
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._max_workers = max_workers or os.cpu_count() or 4
        self._pool: multiprocessing.Pool | None = None
        self._processes: dict[str, ProcessInfo] = {}

    def start(self) -> None:
        """Start the process pool."""
        self._pool = multiprocessing.Pool(processes=self._max_workers)

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_name: str = "",
        **kwargs: Any,
    ) -> str:
        """Submit a task to the pool. Returns a task ID."""
        if self._pool is None:
            raise RuntimeError("ProcessManager not started. Call start() first.")

        task_id = task_name or f"task-{len(self._processes)}"
        info = ProcessInfo(
            name=task_id,
            state=ProcessState.RUNNING,
            started_at=time.time(),
        )
        self._processes[task_id] = info

        result = self._pool.apply_async(fn, args, kwargs)

        try:
            result.get(timeout=0.001)
            info.state = ProcessState.STOPPED
            info.exit_code = 0
        except multiprocessing.TimeoutError:
            pass
        except Exception as exc:
            info.state = ProcessState.FAILED
            info.error = str(exc)

        return task_id

    def get_info(self, task_id: str) -> ProcessInfo | None:
        """Get info about a task."""
        return self._processes.get(task_id)

    def shutdown(self) -> None:
        """Shut down the process pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def list_tasks(self) -> list[ProcessInfo]:
        """List all tracked tasks."""
        return list(self._processes.values())

    @staticmethod
    def get_cpu_count() -> int:
        """Return the number of CPUs available."""
        return os.cpu_count() or 4

    @staticmethod
    def get_pid() -> int:
        """Return the current process ID."""
        return os.getpid()

    @staticmethod
    def send_signal(pid: int, sig: int = signal.SIGTERM) -> None:
        """Send a signal to a process."""
        os.kill(pid, sig)
