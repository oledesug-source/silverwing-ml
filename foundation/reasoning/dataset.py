"""Packed, response-masked reasoning-chain data (M13).

Each example is a structured reasoning trace:

    Problem:
    {question}

    Reasoning:
    1. {step_1}
    2. {step_2}
    ...

    Final Answer:
    {answer}

followed by ``<|endoftext|>``.  This mirrors the legacy 79R structured
reasoning format.  Only the reasoning steps and final answer tokens are
supervised; the problem prompt and format markers are masked with
``IGNORE_INDEX``.  Examples are packed into contiguous ``block_size`` windows
so a window may span example boundaries; the per-token supervision mask
makes this loss-exact.

The trainer uses this dataset with the same masked-CE objective as M11 SFT,
but the richer target format teaches the model to emit a chain of thought
before committing to a final answer.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..tokenizer import TokenizerV2
from .config import (
    EXAMPLE_SEPARATOR,
    FINAL_ANSWER_HEADER,
    PROBLEM_HEADER,
    REASONING_HEADER,
    STEP_PREFIX,
)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ReasoningExample:
    example_id: str
    problem: str
    reasoning_steps: list[str]
    final_answer: str
    reasoning_type: str = "multi_step"
    domain: str = "mathematics"
    difficulty: float = 1.0
    quality_score: float = 1.0

    def format_text(self) -> str:
        """Serialize to the canonical interleaved Problem/Reasoning/Answer text."""
        parts: list[str] = [PROBLEM_HEADER, self.problem, "\n\n", REASONING_HEADER]
        for i, step in enumerate(self.reasoning_steps, start=1):
            if i > 1:
                parts.append("\n")
            parts.append(STEP_PREFIX.format(i=i))
            parts.append(step)
        parts.append(FINAL_ANSWER_HEADER)
        parts.append(self.final_answer)
        parts.append(EXAMPLE_SEPARATOR)
        return "".join(parts)


def split_into_steps(text: str) -> list[str]:
    """Split a paragraph solution into reasoning steps.

    Splits on sentence boundaries (periods, semicolons, and sentence-end markers)
    and groups related sentences into logical steps.
    """
    text = text.strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace or end
    sentences = re.split(r"(?<=[.;])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    steps: list[str] = []
    for sentence in sentences:
        if sentence:
            steps.append(sentence)
    return steps if steps else [text]


def load_reasoning_examples(path: str | Path) -> list[ReasoningExample]:
    records: list[ReasoningExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        problem = record.get("problem", "")
        steps = record.get("reasoning_steps", [])
        final_answer = record.get("final_answer", "")
        if not problem or not final_answer:
            raise ValueError(
                f"line {line_number} of {path} is missing problem or final_answer"
            )
        records.append(
            ReasoningExample(
                example_id=str(record.get("id", f"line-{line_number}")),
                problem=problem,
                reasoning_steps=list(steps),
                final_answer=final_answer,
                reasoning_type=record.get("reasoning_type", "multi_step"),
                domain=record.get("domain", "mathematics"),
                difficulty=float(record.get("difficulty", 1.0)),
                quality_score=float(record.get("quality_score", 1.0)),
            )
        )
    if not records:
        raise ValueError(f"no reasoning examples in {path}")
    return records


def dataset_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tokenize_example(
    example: ReasoningExample, tokenizer: TokenizerV2
) -> tuple[list[int], list[bool]]:
    """Tokenize an example, marking which tokens are supervised.

    Returns (tokens, supervised_mask).  The problem prompt and format markers
    are masked (False); the reasoning steps and final answer are supervised (True).
    """
    full_text = example.format_text()
    tokens = tokenizer.encode(full_text)

    # Compute the token boundary offset: problem header + problem text + separators
    prompt_prefix = f"{PROBLEM_HEADER}{example.problem}\n\n{REASONING_HEADER}"
    prompt_prefix_ids = tokenizer.encode(prompt_prefix)
    # The answer suffix starts after all the reasoning steps; everything from
    # the first reasoning step onwards is supervised.
    supervised = [False] * len(tokens)
    # Everything after the prompt prefix is supervised (reasoning + answer + eos)
    offset = len(prompt_prefix_ids)
    for i in range(max(offset, 0), len(tokens)):
        supervised[i] = True
    return tokens, supervised


class ReasoningDataset:
    def __init__(
        self,
        examples: Iterable[ReasoningExample],
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
        chosen = [
            ex for ex in shuffled if (ex.example_id in eval_ids) == (split == "eval")
        ]
        self.n_examples = len(chosen)
        self.n_documents = len(chosen)

        all_tokens: list[int] = []
        all_supervised: list[bool] = []
        for example in chosen:
            tokens, supervised = _tokenize_example(example, tokenizer)
            if not tokens:
                continue
            all_tokens.extend(tokens)
            all_supervised.extend(supervised)
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
        self.supervised_tokens = sum(all_supervised)

    def __len__(self) -> int:
        return self.n_blocks

    def block(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        xs = self.tokens[start:end]
        ys = [
            token if flag else IGNORE_INDEX
            for token, flag in zip(
                self.tokens[start + 1 : end + 1],
                self.supervised[start + 1 : end + 1],
            )
        ]
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
                yield (
                    xs.reshape(batch_size, self.block_size),
                    ys.reshape(batch_size, self.block_size),
                )
