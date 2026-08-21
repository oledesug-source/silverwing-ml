"""CPU monitoring and process resource tracking for ML workloads.

Provides real-time CPU usage monitoring, thread/process tracking,
and resource limit enforcement.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CpuStats:
    """Snapshot of CPU state."""

    cpu_count: int = 0
    cpu_percent: float = 0.0
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0
    user_time: float = 0.0
    system_time: float = 0.0
    thread_count: int = 0

    @property
    def load_per_core(self) -> float:
        """Average load per CPU core."""
        if self.cpu_count == 0:
            return 0.0
        return self.load_avg_1m / self.cpu_count


@dataclass
class ResourceLimits:
    """Resource limits for a managed workload."""

    max_memory_mb: float = 0.0
    max_cpu_percent: float = 100.0
    max_threads: int = 0
    max_runtime_seconds: float = 0.0


class CpuMonitor:
    """Monitors CPU usage over time.

    Usage::

        monitor = CpuMonitor(interval=0.5)
        monitor.start()

        # ... run workload ...

        stats = monitor.snapshot()
        print(f"CPU: {stats.cpu_percent:.1f}%")

        monitor.stop()
        history = monitor.get_history()
    """

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._history: list[tuple[float, CpuStats]] = []
        self._lock = threading.Lock()
        self._callback: Callable[[CpuStats], None] | None = None

    def start(self) -> None:
        """Start monitoring in background."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def on_sample(self, callback: Callable[[CpuStats], None]) -> None:
        """Register a callback for each CPU sample."""
        self._callback = callback

    def snapshot(self) -> CpuStats:
        """Take an immediate CPU snapshot."""
        return self._sample()

    def get_history(self) -> list[tuple[float, CpuStats]]:
        """Get collected CPU history."""
        with self._lock:
            return list(self._history)

    def get_average(self, last_n: int = 10) -> float:
        """Get average CPU percent over the last N samples."""
        with self._lock:
            samples = self._history[-last_n:]
        if not samples:
            return 0.0
        return sum(s[1].cpu_percent for s in samples) / len(samples)

    def _sample(self) -> CpuStats:
        """Take a single CPU sample."""
        stats = CpuStats(cpu_count=os.cpu_count() or 1)

        try:
            import psutil
            stats.cpu_percent = psutil.cpu_percent(interval=0.1)
            load = os.getloadavg()
            stats.load_avg_1m = load[0]
            stats.load_avg_5m = load[1]
            stats.load_avg_15m = load[2]
            proc = psutil.Process()
            cpu_times = proc.cpu_times()
            stats.user_time = cpu_times.user
            stats.system_time = cpu_times.system
            stats.thread_count = proc.num_threads()
        except (ImportError, OSError, AttributeError):
            try:
                load = os.getloadavg()
                stats.load_avg_1m = load[0]
                stats.load_avg_5m = load[1]
                stats.load_avg_15m = load[2]
            except (OSError, AttributeError):
                pass

        return stats

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            stats = self._sample()
            now = time.time()
            with self._lock:
                self._history.append((now, stats))
                if len(self._history) > 1000:
                    self._history.pop(0)
            if self._callback:
                self._callback(stats)
            self._stop_event.wait(self._interval)


class ResourceTracker:
    """Tracks resource consumption and enforces limits.

    Usage::

        tracker = ResourceTracker(limits=ResourceLimits(
            max_memory_mb=4096,
            max_cpu_percent=80.0,
            max_runtime_seconds=3600,
        ))
        tracker.start()

        if tracker.should_stop():
            print("Resource limit exceeded!")
            tracker.stop()
    """

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self._limits = limits or ResourceLimits()
        self._start_time: float | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._violations: list[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start tracking."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop tracking."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def should_stop(self) -> bool:
        """Check if any resource limit has been exceeded."""
        with self._lock:
            return len(self._violations) > 0

    def get_violations(self) -> list[str]:
        """Get all recorded resource limit violations."""
        with self._lock:
            return list(self._violations)

    def elapsed_seconds(self) -> float:
        """Seconds since tracking started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _check_loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_limits()
            if self.should_stop():
                break
            self._stop_event.wait(0.1)

    def _check_limits(self) -> None:
        limits = self._limits

        if limits.max_runtime_seconds > 0 and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > limits.max_runtime_seconds:
                with self._lock:
                    self._violations.append(
                        f"Runtime exceeded: {elapsed:.1f}s > {limits.max_runtime_seconds}s"
                    )

        if limits.max_cpu_percent < 100.0:
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                if cpu > limits.max_cpu_percent:
                    with self._lock:
                        self._violations.append(
                            f"CPU exceeded: {cpu:.1f}% > {limits.max_cpu_percent}%"
                        )
            except ImportError:
                pass
