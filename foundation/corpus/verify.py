"""Dataset integrity verification.

Recomputes the dataset root digest from shard files on disk and compares it to
the digest recorded in the manifest, so corruption, truncation or tampering in
a released dataset is detected before training consumes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import dataset_root_digest, split_digest
from .storage import SHARD_SIZE


@dataclass
class VerificationResult:
    manifest_path: Path
    ok: bool = False
    missing_shards: list[str] = field(default_factory=list)
    split_errors: list[str] = field(default_factory=list)
    recorded_dataset_hash: str = ""
    computed_dataset_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "manifest_path": str(self.manifest_path),
            "missing_shards": self.missing_shards,
            "split_errors": self.split_errors,
            "recorded_dataset_hash": self.recorded_dataset_hash,
            "computed_dataset_hash": self.computed_dataset_hash,
        }


def verify_dataset(output_dir, expected_dataset_hash: str | None = None) -> VerificationResult:
    """Verify a sharded dataset against its manifest.

    Optionally compare against an expected dataset hash (e.g. the hash pinned
    in an experiment manifest), which catches release substitution.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    result = VerificationResult(manifest_path=manifest_path)
    if not manifest_path.exists():
        result.split_errors.append("manifest.json missing")
        return result

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    shard_paths: dict[str, list[Path]] = {}
    for split_name, info in splits.items():
        for shard_name in info.get("shards", []):
            shard_path = output_dir / shard_name
            if not shard_path.exists():
                result.missing_shards.append(shard_name)
            else:
                shard_paths.setdefault(split_name, []).append(shard_path)

    if result.missing_shards:
        result.split_errors.append("missing shard files detected")
    for split_name, paths in shard_paths.items():
        records = splits[split_name]["records"]
        expected_shards = records // SHARD_SIZE + (1 if records % SHARD_SIZE else 0)
        if len(paths) != expected_shards:
            result.split_errors.append(f"{split_name}: shard count mismatch with record count")

    split_digests = {name: split_digest(paths) for name, paths in shard_paths.items() if paths}
    result.computed_dataset_hash = dataset_root_digest(split_digests)
    result.recorded_dataset_hash = manifest.get("dataset_hash", "")

    if result.recorded_dataset_hash and result.computed_dataset_hash != result.recorded_dataset_hash:
        result.split_errors.append("dataset_hash mismatch: manifest != computed")
    if expected_dataset_hash is not None and expected_dataset_hash != result.recorded_dataset_hash:
        result.split_errors.append("dataset_hash mismatch: pinned expected hash != manifest")

    result.ok = not result.missing_shards and not result.split_errors
    return result
