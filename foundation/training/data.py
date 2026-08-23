"""Tokenized, packed, causal sequence data for pretraining.

Reads the sharded corpus (train.*.jsonl / validation.*.jsonl) produced by the
M02/M03 pipeline, tokenizes documents with Tokenizer V2, appends the
<|endoftext|> separator after every document and packs the token stream into
contiguous block_size windows. Targets are the input shifted by one position,
so each block models the exact token stream seen by the decoder.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from ..tokenizer import TokenizerV2


class ShuffledBatchStream:
    """Infinite deterministic stream with serializable shuffle progress.

    Saving this state immediately after a training step lets a resumed run use
    the exact next batch rather than merely restarting a seeded shuffle.
    """

    def __init__(self, data: PretrainingData, batch_size: int, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.data = data
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.indices: list[int] = []
        self.position = 0

    def __iter__(self) -> ShuffledBatchStream:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data.n_blocks == 0:
            raise ValueError("cannot stream batches from empty training data")
        if self.position >= len(self.indices):
            self.indices = self.data.shuffled_indices(self.rng)
            self.position = 0
        batch_indices = self.indices[self.position : self.position + self.batch_size]
        self.position += len(batch_indices)
        return self.data.batch(batch_indices)

    def state_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "rng_state": self.rng.getstate(),
            "indices": list(self.indices),
            "position": self.position,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise ValueError("checkpoint batch_size does not match the configured batch_size")
        indices = [int(index) for index in state.get("indices", [])]
        position = int(state.get("position", 0))
        if any(index < 0 or index >= self.data.n_blocks for index in indices):
            raise ValueError("checkpoint data stream contains an invalid block index")
        if position < 0 or position > len(indices):
            raise ValueError("checkpoint data stream has an invalid position")
        if "rng_state" not in state:
            raise ValueError("checkpoint data stream is missing rng_state")
        self.rng.setstate(state["rng_state"])
        self.indices = indices
        self.position = position


class PretrainingData:
    def __init__(
        self,
        corpus_dir: str | Path,
        tokenizer: TokenizerV2,
        split: str = "train",
        block_size: int = 512,
        max_tokens: int | None = None,
    ) -> None:
        self.block_size = block_size
        self.eos_id = tokenizer.special_ids["<|endoftext|>"]
        corpus_dir = Path(corpus_dir)
        cache_path = corpus_dir / f".token-cache-{tokenizer.digest()[:12]}-{split}.pt"
        cached = self._load_cache(cache_path)
        if cached is not None:
            self.tokens, self.n_documents = cached
        else:
            tokens: list[int] = []
            n_documents = 0
            for shard in sorted(corpus_dir.glob(f"{split}.*.jsonl")):
                for line in shard.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get("text", "")
                    if not text:
                        continue
                    tokens.extend(tokenizer.encode(text))
                    tokens.append(self.eos_id)
                    n_documents += 1
                    if max_tokens is not None and len(tokens) >= max_tokens:
                        break
                if max_tokens is not None and len(tokens) >= max_tokens:
                    break
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            self.n_documents = n_documents
            self._save_cache(cache_path, self.tokens, n_documents)
        self.n_blocks = max(0, (self.num_tokens() - 1) // block_size)

    @staticmethod
    def _load_cache(path: Path) -> tuple[torch.Tensor, int] | None:
        """Load a previously tokenized stream; None on any mismatch/corruption."""
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            tokens = payload["tokens"]
            if tokens.dtype not in (torch.long, torch.int32, torch.int16) or tokens.dim() != 1:
                return None
            return tokens, int(payload["n_documents"])
        except Exception:
            return None

    @staticmethod
    def _save_cache(path: Path, tokens: torch.Tensor, n_documents: int) -> None:
        """Best-effort cache write; failures never block training."""
        try:
            tmp = path.with_name(path.name + ".tmp")
            torch.save({"tokens": tokens, "n_documents": int(n_documents)}, tmp)
            tmp.replace(path)
        except OSError:
            pass

    def __len__(self) -> int:
        return self.n_blocks

    def num_tokens(self) -> int:
        return int(self.tokens.numel())

    def _block(self, index: int) -> torch.Tensor:
        start = index * self.block_size
        return self.tokens[start : start + self.block_size + 1]

    def batch(self, block_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not block_indices:
            raise ValueError("cannot build a batch from empty block indices")
        starts = torch.as_tensor(block_indices, dtype=torch.long) * self.block_size
        offsets = torch.arange(self.block_size + 1)
        seq = self.tokens[starts[:, None] + offsets[None, :]]
        # .long() is a no-op view for long caches and an upcast for int32/16 ones
        return seq[:, :-1].long(), seq[:, 1:].long()

    def shuffled_indices(self, rng: random.Random) -> list[int]:
        indices = list(range(self.n_blocks))
        rng.shuffle(indices)
        return indices

    def batches(self, batch_size: int, seed: int):
        """Infinite shuffled stream of (input_ids, targets) batches."""
        yield from ShuffledBatchStream(self, batch_size, seed)

    def batch_stream(self, batch_size: int, seed: int) -> ShuffledBatchStream:
        return ShuffledBatchStream(self, batch_size, seed)

    def ordered_batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.n_blocks == 0:
            return None
        return self.batch(list(range(min(n, self.n_blocks))))
