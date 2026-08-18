"""Preference data loading for DPO (M12).

Each line of a preference dataset is a JSON object with at minimum:
  - ``instruction``: the shared prompt shown to both responses
  - ``chosen``: the preferred (winner) response
  - ``rejected``: the dispreferred (loser) response

The dataset packs preference pairs into blocks of ``block_size`` tokens.
Each block contains one prompt+chosen and one prompt+rejected sequence,
padded to the same length. Labels for log-prob computation use IGNORE_INDEX
for padding positions.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from ..tokenizer import TokenizerV2

IGNORE_INDEX = -100

QUESTION_PREFIX = "Question: "
ANSWER_PREFIX = "Answer: "


@dataclass(frozen=True)
class PreferenceExample:
    example_id: str
    instruction: str
    chosen: str
    rejected: str

    def format_prompt(self) -> str:
        return f"{QUESTION_PREFIX}{self.instruction}\n{ANSWER_PREFIX}"

    def format_chosen(self) -> str:
        return f"{self.format_prompt()}{self.chosen}<|endoftext|>"

    def format_rejected(self) -> str:
        return f"{self.format_prompt()}{self.rejected}<|endoftext|>"


def load_preferences(path: str | Path) -> list[PreferenceExample]:
    records: list[PreferenceExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        instruction = record.get("instruction", "")
        chosen = record.get("chosen", record.get("response_win", ""))
        rejected = record.get("rejected", record.get("response_rej", ""))
        if not instruction or not chosen or not rejected:
            raise ValueError(
                f"line {line_number} of {path} is missing instruction/chosen/rejected"
            )
        records.append(
            PreferenceExample(
                example_id=str(record.get("id", f"line-{line_number}")),
                instruction=instruction,
                chosen=chosen,
                rejected=rejected,
            )
        )
    if not records:
        raise ValueError(f"no preference examples in {path}")
    return records


def dataset_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _encode_pair(
    example: PreferenceExample,
    tokenizer: TokenizerV2,
    block_size: int,
) -> tuple[list[int], list[int], int]:
    """Tokenize a (prompt, chosen/rejected) pair.

    Returns (input_ids, labels, prompt_len) where labels is aligned to
    input_ids with IGNORE_INDEX for padding positions and includes the
    shifted-by-one target convention used by causal LM log-prob.
    """
    eos_id = tokenizer.special_ids["<|endoftext|>"]
    prompt_ids = tokenizer.encode(example.format_prompt())
    chosen_ids = tokenizer.encode(example.chosen) + [eos_id]
    rejected_ids = tokenizer.encode(example.rejected) + [eos_id]

    prompt_len = len(prompt_ids)
    chosen_full = prompt_ids + chosen_ids
    rejected_full = prompt_ids + rejected_ids

    if len(chosen_full) > block_size:
        chosen_full = chosen_full[:block_size]
    if len(rejected_full) > block_size:
        rejected_full = rejected_full[:block_size]

    return chosen_full, rejected_full, prompt_len


class PreferenceDataset:
    """Packs preference pairs into blocks of block_size tokens.

    Each block yields (input_ids_w, labels_w, input_ids_l, labels_l, prompt_len)
    where the chosen and rejected sequences are padded to the same length.
    Labels use IGNORE_INDEX for padding and for prompt tokens (only response
    tokens are supervised).
    """

    def __init__(
        self,
        examples: list[PreferenceExample],
        tokenizer: TokenizerV2,
        block_size: int,
        seed: int,
        eval_fraction: float,
        split: str,
    ) -> None:
        self.block_size = block_size
        self.tokenizer = tokenizer
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

        self.pairs: list[tuple[list[int], list[int], int]] = []
        for ex in chosen:
            w_ids, l_ids, plen = _encode_pair(ex, tokenizer, block_size)
            self.pairs.append((w_ids, l_ids, plen))

        self.n_blocks = len(self.pairs)

    def __len__(self) -> int:
        return self.n_blocks

    def block(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (input_ids_w, labels_w, input_ids_l, labels_l) padded to block_size."""
        w_ids, l_ids, plen = self.pairs[index]
        pad_id = self.tokenizer.special_ids["<|endoftext|>"]
        max_len = max(len(w_ids), len(l_ids))
        max_len = min(max_len, self.block_size)

        def _pad(ids: list[int]) -> tuple[list[int], list[int]]:
            labels = [IGNORE_INDEX] * len(ids)
            for i in range(plen, len(ids)):
                labels[i] = ids[i]
            pad_count = max_len - len(ids)
            ids = ids + [pad_id] * pad_count
            labels = labels + [IGNORE_INDEX] * pad_count
            return ids, labels

        w_ids_padded, w_labels = _pad(w_ids)
        l_ids_padded, l_labels = _pad(l_ids)
        # Pad to block_size
        pad_count = self.block_size - max_len
        w_ids_padded += [pad_id] * pad_count
        w_labels += [IGNORE_INDEX] * pad_count
        l_ids_padded += [pad_id] * pad_count
        l_labels += [IGNORE_INDEX] * pad_count
        return (
            torch.tensor(w_ids_padded, dtype=torch.long),
            torch.tensor(w_labels, dtype=torch.long),
            torch.tensor(l_ids_padded, dtype=torch.long),
            torch.tensor(l_labels, dtype=torch.long),
        )

    def ordered_blocks(
        self, n: int
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.n_blocks == 0:
            raise ValueError("cannot build an ordered batch from an empty dataset")
        return [self.block(i) for i in range(min(n, self.n_blocks))]

    def shuffled_indices(self, rng: random.Random) -> list[int]:
        indices = list(range(self.n_blocks))
        rng.shuffle(indices)
        return indices

    def shuffled_batches(self, batch_size: int, seed: int):
        rng = random.Random(seed)
        while True:
            indices = self.shuffled_indices(rng)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if len(batch) < batch_size:
                    break
                yield [self.block(i) for i in batch]
