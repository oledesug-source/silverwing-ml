"""Cryptographic utilities for data integrity and hashing.

Provides checksums, file hashing, and tamper detection for datasets,
models, and experiment artifacts.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Checksum:
    """A checksum record for integrity verification."""

    algorithm: str
    digest: str
    size_bytes: int = 0

    def verify(self, data: bytes) -> bool:
        """Verify data against this checksum."""
        h = hashlib.new(self.algorithm)
        h.update(data)
        return hmac.compare_digest(h.hexdigest(), self.digest)


class Hasher:
    """Multi-algorithm hasher for files and data.

    Usage::

        hasher = Hasher()
        hasher.update(b"hello world")
        result = hasher.digest()
        print(result["sha256"])

        # File hashing
        digest = Hasher.hash_file("model.pt")
    """

    ALGORITHMS = ("md5", "sha1", "sha256", "sha512")

    def __init__(self, algorithms: tuple[str, ...] = ("sha256",)) -> None:
        self._hashers: dict[str, hashlib._Hash] = {}
        for algo in algorithms:
            if algo not in self.ALGORITHMS:
                raise ValueError(f"Unknown algorithm: {algo}")
            self._hashers[algo] = hashlib.new(algo)

    def update(self, data: bytes) -> None:
        """Feed data into all hashers."""
        for h in self._hashers.values():
            h.update(data)

    def digest(self) -> dict[str, str]:
        """Return hex digests for all algorithms."""
        return {name: h.hexdigest() for name, h in self._hashers.items()}

    def reset(self) -> None:
        """Reset all hashers."""
        for h in self._hashers.values():
            h.reset()

    @staticmethod
    def hash_file(
        path: str | Path,
        algorithms: tuple[str, ...] = ("sha256",),
        chunk_size: int = 8192,
    ) -> dict[str, str]:
        """Hash a file with multiple algorithms."""
        hashers = {algo: hashlib.new(algo) for algo in algorithms}
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                for h in hashers.values():
                    h.update(chunk)
        return {name: h.hexdigest() for name, h in hashers.items()}

    @staticmethod
    def hash_string(
        text: str,
        algorithms: tuple[str, ...] = ("sha256",),
    ) -> dict[str, str]:
        """Hash a string with multiple algorithms."""
        data = text.encode("utf-8")
        return {algo: hashlib.new(algo, data).hexdigest() for algo in algorithms}

    @staticmethod
    def generate_salt(length: int = 32) -> str:
        """Generate a cryptographically secure random salt."""
        return secrets.token_hex(length)
