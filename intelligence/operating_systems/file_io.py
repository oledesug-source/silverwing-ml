"""File system I/O manager with buffering, caching, and async patterns.

Provides ML-specific file I/O abstractions for large model files,
dataset streaming, and checkpoint management.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileStats:
    """Statistics for a file or directory."""

    path: str
    size_bytes: int = 0
    is_dir: bool = False
    file_count: int = 0
    modified_at: float = 0.0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 * 1024 * 1024)


class FileCache:
    """In-memory LRU cache for frequently accessed files.

    Usage::

        cache = FileCache(max_size_mb=256)
        data = cache.read("model_config.json")
        cache.write("model_config.json", new_data)
    """

    def __init__(self, max_size_mb: float = 256.0, ttl_seconds: float = 300.0) -> None:
        self._max_bytes = int(max_size_mb * 1024 * 1024)
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._current_size = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def read(self, path: str | Path) -> bytes | None:
        """Read from cache, or None if not cached."""
        key = str(path)
        with self._lock:
            if key in self._cache:
                data, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._hits += 1
                    return data
                self._evict(key)
            self._misses += 1
        return None

    def write(self, path: str | Path, data: bytes) -> None:
        """Write data to cache."""
        key = str(path)
        data_len = len(data)
        with self._lock:
            if key in self._cache:
                self._evict(key)
            while self._current_size + data_len > self._max_bytes and self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                self._evict(oldest_key)
            self._cache[key] = (data, time.time())
            self._current_size += data_len

    def invalidate(self, path: str | Path) -> bool:
        """Remove an entry from the cache."""
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._evict(key)
                return True
        return False

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def _evict(self, key: str) -> None:
        if key in self._cache:
            data, _ = self._cache.pop(key)
            self._current_size -= len(data)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "size_mb": round(self._current_size / (1024 * 1024), 2),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total:.1%}" if total else "0%",
        }


class SafeWriter:
    """Atomic file writer that prevents corruption on crash.

    Usage::

        with SafeWriter("model.pt") as tmp_path:
            torch.save(model.state_dict(), tmp_path)
        # File is atomically moved on exit
    """

    def __init__(self, target: str | Path, suffix: str = ".tmp") -> None:
        self._target = Path(target)
        self._suffix = suffix
        self._tmp_path: Path | None = None

    @contextmanager
    def write(self) -> Generator[Path, None, None]:
        """Context manager that yields a temp path and atomically moves on success."""
        self._target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            suffix=self._suffix, dir=str(self._target.parent)
        )
        os.close(fd)
        self._tmp_path = Path(tmp)
        try:
            yield self._tmp_path
            shutil.move(str(self._tmp_path), str(self._target))
        except BaseException:
            if self._tmp_path.exists():
                self._tmp_path.unlink()
            raise
        finally:
            self._tmp_path = None

    def __enter__(self) -> Path:
        # Pre-create the generator for direct usage
        self._ctx = self.write()
        return self._ctx.__enter__()

    def __exit__(self, *args: Any) -> None:
        self._ctx.__exit__(*args)


class FileManager:
    """Manages ML files: streaming, checksums, directory stats, safe writes.

    Usage::

        fm = FileManager(cache_size_mb=512)
        stats = fm.dir_stats("datasets/")
        print(f"Total: {stats['total_size_mb']:.1f} MB")
    """

    def __init__(self, cache_size_mb: float = 256.0) -> None:
        self._cache = FileCache(max_size_mb=cache_size_mb)

    @property
    def cache(self) -> FileCache:
        return self._cache

    def dir_stats(self, path: str | Path) -> dict[str, Any]:
        """Compute directory statistics."""
        p = Path(path)
        if not p.exists():
            return {"path": str(p), "exists": False}

        total_size = 0
        file_count = 0
        dir_count = 0
        if p.is_file():
            total_size = p.stat().st_size
            file_count = 1
        else:
            for item in p.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
                elif item.is_dir():
                    dir_count += 1

        return {
            "path": str(p),
            "exists": True,
            "is_dir": p.is_dir(),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
            "file_count": file_count,
            "dir_count": dir_count,
        }

    def checksum_file(
        self, path: str | Path, algorithm: str = "sha256"
    ) -> str:
        """Compute file checksum."""
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def stream_file(
        self,
        path: str | Path,
        chunk_size: int = 65536,
    ) -> Generator[bytes, None, None]:
        """Stream a file in chunks for memory-efficient processing."""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def ensure_dir(self, path: str | Path) -> Path:
        """Create directory tree if it doesn't exist."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def safe_write(self, target: str | Path) -> SafeWriter:
        """Get a SafeWriter for atomic file writes."""
        return SafeWriter(target)
