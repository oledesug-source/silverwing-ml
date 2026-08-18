"""Tokenizer V2: byte-level BPE with special tokens, save/load and hashing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from .bpe import (
    MERGE_BASE,
    NUM_SPECIALS,
    SPECIAL_TOKENS,
    build_ranks,
    encode_with_merges,
)

TOKENIZER_VERSION = "tokenizer-v1"


class TokenizerV2:
    def __init__(
        self,
        merges: list[tuple[str, str]],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        version: str = TOKENIZER_VERSION,
    ) -> None:
        if len(special_tokens) != NUM_SPECIALS:
            raise ValueError(f"Expected {NUM_SPECIALS} special tokens, got {len(special_tokens)}")
        self.special_tokens = special_tokens
        self.version = version
        self.merges = list(merges)
        self.pair_rank, self.token_rank = build_ranks(self.merges)
        self.special_ids = {tok: i for i, tok in enumerate(special_tokens)}
        self._special_re = re.compile("|".join(re.escape(tok) for tok in special_tokens))

    @property
    def vocab_size(self) -> int:
        return MERGE_BASE + len(self.merges)

    def vocab(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for i, token in enumerate(self.special_tokens):
            mapping[token] = i
        for b in range(256):
            mapping[chr(b)] = b + NUM_SPECIALS
        for rank, (a, b) in enumerate(self.merges):
            mapping[a + b] = MERGE_BASE + rank
        return mapping

    def encode(self, text: str) -> list[int]:
        """Encode text to ids, preserving special tokens as their own ids."""
        ids: list[int] = []
        position = 0
        for match in self._special_re.finditer(text):
            if match.start() > position:
                ids.extend(encode_with_merges(text[position : match.start()], self.pair_rank, self.token_rank))
            ids.append(self.special_ids[match.group()])
            position = match.end()
        if position < len(text):
            ids.extend(encode_with_merges(text[position:], self.pair_rank, self.token_rank))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to text (special tokens decode to their literal string)."""
        vocab = self.vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        token_strings = [id_to_token.get(int(token_id), "") for token_id in ids]
        return "".join(token_strings).encode("latin-1").decode("utf-8", errors="replace")

    def digest(self) -> str:
        payload = json.dumps(self.vocab(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "version": self.version,
            "algorithm": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "num_specials": NUM_SPECIALS,
        }
        (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "merges.json").write_text(
            json.dumps([list(m) for m in self.merges], ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "vocab.json").write_text(
            json.dumps(self.vocab(), ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "tokenizer_hash").write_text(self.digest(), encoding="utf-8")
        return output_dir

    @classmethod
    def load(cls, input_dir: str | Path, version: Optional[str] = None) -> "TokenizerV2":
        input_dir = Path(input_dir)
        config_path = input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in json.loads((input_dir / "merges.json").read_text(encoding="utf-8"))]
        specials = tuple(config.get("special_tokens", SPECIAL_TOKENS))
        instance = cls(merges=merges, special_tokens=specials, version=config.get("version", TOKENIZER_VERSION))
        if version is not None and instance.version != version:
            raise ValueError(f"Tokenizer version {instance.version} does not match required {version}")
        return instance
