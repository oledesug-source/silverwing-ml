"""Sharded JSONL storage for the processed corpus.

Writes records into numbered shard files plus a manifest that records
per-split record counts, token counts, content hashes and a dataset root
digest so dataset releases are auditable and their integrity verifiable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hashing import dataset_root_digest, split_digest
from .schema import DocumentRecord, Split

SHARD_SIZE = 10_000


class ShardWriter:
    def __init__(self, output_dir: Path, dataset_name: str = "silverwing") -> None:
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

    def write(self, split_records: dict[str, list[DocumentRecord]]) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "dataset_name": self.dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "splits": {},
            "shards": [],
        }
        split_shards: dict[str, list[Path]] = {}
        for split in (Split.TRAIN.value, Split.VALIDATION.value, Split.TEST.value):
            records = split_records.get(split, [])
            info, shard_paths = self._write_split(records, split)
            manifest["splits"][split] = info
            split_shards[split] = shard_paths
        split_digests = {split: split_digest(paths) for split, paths in split_shards.items() if paths}
        manifest["dataset_hash"] = dataset_root_digest(split_digests)
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _write_split(self, records: Iterable[DocumentRecord], split: str) -> tuple[dict, list[Path]]:
        token_total = 0
        count = 0
        hashes: list[str] = []
        buffer: list[str] = []
        shard_paths: list[Path] = []
        shard_index = 0

        def flush() -> None:
            nonlocal shard_index
            if not buffer:
                return
            shard_name = f"{split}.{shard_index}.jsonl"
            shard_path = self.output_dir / shard_name
            shard_path.write_text("\n".join(buffer) + "\n", encoding="utf-8")
            shard_paths.append(shard_path)
            buffer.clear()
            shard_index += 1

        for record in records:
            token_total += record.token_count
            hashes.append(record.content_hash)
            buffer.append(json.dumps(record.to_dict(), ensure_ascii=False))
            count += 1
            if len(buffer) >= SHARD_SIZE:
                flush()
        flush()
        return (
            {
                "records": count,
                "tokens": token_total,
                "first_content_hash": hashes[0] if hashes else None,
                "last_content_hash": hashes[-1] if hashes else None,
                "shards": [path.name for path in shard_paths],
            },
            shard_paths,
        )
