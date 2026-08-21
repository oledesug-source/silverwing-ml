"""Memory management utilities for ML workloads.

Tracks memory usage, provides OOM-safe allocation patterns,
and monitors GPU/CPU memory during training.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryStats:
    """Snapshot of current memory usage."""

    rss_bytes: int = 0
    vms_bytes: int = 0
    available_bytes: int = 0
    used_percent: float = 0.0
    gpu_allocated_bytes: int = 0
    gpu_reserved_bytes: int = 0

    @property
    def rss_mb(self) -> float:
        """Resident set size in megabytes."""
        return self.rss_bytes / (1024 * 1024)

    @property
    def vms_mb(self) -> float:
        """Virtual memory size in megabytes."""
        return self.vms_bytes / (1024 * 1024)

    @property
    def available_mb(self) -> float:
        """Available system memory in megabytes."""
        return self.available_bytes / (1024 * 1024)


class MemoryManager:
    """Monitors and manages memory for ML training/inference.

    Usage::

        mm = MemoryManager()
        stats = mm.snapshot()
        print(f"RSS: {stats.rss_mb:.1f} MB")

        with mm.track("forward_pass"):
            output = model(input_tensor)
    """

    def __init__(self, warn_threshold: float = 0.85) -> None:
        self._warn_threshold = warn_threshold
        self._tracking: dict[str, float] = {}
        self._lock = threading.Lock()

    def snapshot(self) -> MemoryStats:
        """Take a snapshot of current memory usage."""
        stats = MemoryStats()

        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            stats.rss_bytes = mem.rss
            stats.vms_bytes = mem.vms
            stats.available_bytes = psutil.virtual_memory().available
            stats.used_percent = psutil.virtual_memory().percent
        except ImportError:
            stats.rss_bytes = self._get_rss_fallback()

        try:
            import torch
            if torch.cuda.is_available():
                stats.gpu_allocated_bytes = torch.cuda.memory_allocated()
                stats.gpu_reserved_bytes = torch.cuda.memory_reserved()
        except ImportError:
            pass

        return stats

    def check_available(self, needed_bytes: int) -> bool:
        """Check if enough memory is available."""
        stats = self.snapshot()
        return stats.available_bytes > needed_bytes

    def warn_if_high(self) -> str | None:
        """Return a warning message if memory usage is high."""
        stats = self.snapshot()
        if stats.used_percent / 100.0 > self._warn_threshold:
            return (
                f"Memory usage at {stats.used_percent:.1f}% "
                f"({stats.rss_mb:.0f} MB RSS, {stats.available_mb:.0f} MB available)"
            )
        return None

    def track(self, label: str) -> _MemoryTracker:
        """Context manager to track memory delta for a block."""
        return _MemoryTracker(self, label)

    def _get_rss_fallback(self) -> int:
        """Fallback RSS measurement using /proc on Linux."""
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except (FileNotFoundError, ValueError):
            pass
        return 0


class _MemoryTracker:
    """Context manager for tracking memory usage of a code block."""

    def __init__(self, manager: MemoryManager, label: str) -> None:
        self._manager = manager
        self._label = label
        self._start: float = 0
        self._delta_mb: float = 0.0

    def __enter__(self) -> _MemoryTracker:
        stats = self._manager.snapshot()
        self._start = stats.rss_bytes
        return self

    def __exit__(self, *args: Any) -> None:
        stats = self._manager.snapshot()
        delta = stats.rss_bytes - self._start
        self._delta_mb = delta / (1024 * 1024)
        with self._manager._lock:
            self._manager._tracking[self._label] = self._delta_mb

    @property
    def delta_mb(self) -> float:
        """Memory delta in MB after exiting the context."""
        return self._delta_mb
