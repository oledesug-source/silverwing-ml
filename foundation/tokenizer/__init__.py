"""Silverwing Tokenizer V2 — byte-level BPE (GPT-2 style).

Replaces the legacy toy tokenizer (lesson66R). Trained on the corpus-v1
shards, produces vocab.json / merges.json / config.json / tokenizer_hash.
"""

from .bpe import SPECIAL_TOKENS
from .tokenizer import TokenizerV2
from .train import iter_corpus_texts, train_tokenizer_from_corpus

__all__ = [
    "TokenizerV2",
    "SPECIAL_TOKENS",
    "iter_corpus_texts",
    "train_tokenizer_from_corpus",
]
