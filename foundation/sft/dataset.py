"""Packed, response-masked SFT data (M11).

Each example is formatted as ``Question: {instruction}\\nAnswer: {response}``
followed by <|endoftext|>. Only the answer tokens (and the terminator) are
supervised; instruction, format markers and padding are masked out so the
model learns to continue after the question rather than reproduce it.
Examples are packed into contiguous block_size windows, so a window may span
example boundaries; the per-token supervision mask makes this loss-exact.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..tokenizer import TokenizerV2

QUESTION_PREFIX = "Question: "
ANSWER_PREFIX = "Answer: "
EOS_MARKER = "<|endoftext|>"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SftExample:
    example_id: str
    instruction: str
    response: str

    def format_text(self) -> str:
        return f"{QUESTION_PREFIX}{self.instruction}\n{ANSWER_PREFIX}{self.response}"


def load_examples(path: str | Path) -> list[SftExample]:
    records: list[SftExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        instruction = record.get("instruction", "")
        response = record.get("response", "")
        if not instruction or not response:
            raise ValueError(f"line {line_number} of {path} is missing instruction or response")
        records.append(
            SftExample(
                example_id=str(record.get("id", f"line-{line_number}")),
                instruction=instruction,
                response=response,
            )
        )
    if not records:
        raise ValueError(f"no SFT examples in {path}")
    return records


def dataset_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tokenize_example(example: SftExample, tokenizer: TokenizerV2) -> tuple[list[int], list[bool]]:
    tokens = tokenizer.encode(example.format_text())
    answer_start = len(tokenizer.encode(f"{QUESTION_PREFIX}{example.instruction}\n{ANSWER_PREFIX}"))
    supervised = [False] * len(tokens)
    for i in range(answer_start, len(tokens)):
        supervised[i] = True
    return tokens, supervised


class SftDataset:
    def __init__(
        self,
        examples: Iterable[SftExample],
        tokenizer: TokenizerV2,
        block_size: int,
        seed: int,
        eval_fraction: float,
        split: str,
    ) -> None:
        self.block_size = block_size
        example_list = list(examples)
        rng = random.Random(seed)
        shuffled = list(example_list)
        rng.shuffle(shuffled)
        n_eval = round(len(shuffled) * eval_fraction)
        eval_ids = {ex.example_id for ex in shuffled[:n_eval]}
        chosen = [ex for ex in shuffled if (ex.example_id in eval_ids) == (split == "eval")]
        self.n_examples = len(chosen)
        self.n_documents = len(chosen)

        all_tokens: list[int] = []
        all_supervised: list[bool] = []
        eos_id = tokenizer.special_ids["<|endoftext|>"]
        for example in chosen:
            tokens, supervised = _tokenize_example(example, tokenizer)
            if not tokens:
                continue
            all_tokens.extend(tokens)
            all_supervised.extend(supervised)
            all_tokens.append(eos_id)
            all_supervised.append(True)
        self.supervised_tokens = sum(all_supervised)

        pad = tokenizer.special_ids["<|endoftext|>"]
        if len(all_tokens) % block_size:
            pad_count = block_size - len(all_tokens) % block_size
            all_tokens.extend([pad] * pad_count)
            all_supervised.extend([False] * pad_count)
        all_tokens.append(pad)
        all_supervised.append(False)
        self.tokens = all_tokens
        self.supervised = all_supervised
        self.n_blocks = (len(all_tokens) - 1) // block_size if all_tokens else 0

    def __len__(self) -> int:
        return self.n_blocks

    def block(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        xs = self.tokens[start:end]
        ys = [token if flag else IGNORE_INDEX for token, flag in zip(self.tokens[start + 1 : end + 1], self.supervised[start + 1 : end + 1])]
        return (
            torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
        )

    def ordered_blocks(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_blocks == 0:
            raise ValueError("cannot build an ordered batch from an empty dataset")
        xs, ys = self.block(0)
        for i in range(1, min(n, self.n_blocks)):
            bx, by = self.block(i)
            xs = torch.cat([xs, bx])
            ys = torch.cat([ys, by])
        return xs.reshape(-1, self.block_size), ys.reshape(-1, self.block_size)

    def shuffled_batches(self, batch_size: int, seed: int):
        rng = random.Random(seed)
        indices = list(range(self.n_blocks))
        rng.shuffle(indices)
        while True:
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if len(batch) < batch_size:
                    break
                xs, ys = self.block(batch[0])
                for i in batch[1:]:
                    bx, by = self.block(i)
                    xs = torch.cat([xs, bx])
                    ys = torch.cat([ys, by])
                yield xs.reshape(batch_size, self.block_size), ys.reshape(batch_size, self.block_size)
