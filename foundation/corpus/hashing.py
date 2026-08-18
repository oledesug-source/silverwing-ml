"""Streaming hashing helpers and dataset-level digests.

The dataset hash is a root digest over the raw bytes of every shard file, so
any byte-level corruption in a released dataset is detectable. Hashing is
streamed to stay memory-safe at corpus scale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_digest(shard_files: list[Path]) -> str:
    """Digest over the concatenated bytes of a split's shard files."""
    digest = hashlib.sha256()
    for shard in shard_files:
        digest.update(stream_sha256(shard).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def dataset_root_digest(split_digests: dict[str, str]) -> str:
    """Deterministic root digest over named split digests."""
    digest = hashlib.sha256()
    for name in sorted(split_digests):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(split_digests[name].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
