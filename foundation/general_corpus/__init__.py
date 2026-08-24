"""General-conversation corpus (M17) - deterministic Q/A bank."""

from .bank import (
    BANK,
    BAD_RESPONSES,
    BankItem,
    expand_bank,
    general_preference_pairs,
    write_jsonl,
)

__all__ = [
    "BANK",
    "BAD_RESPONSES",
    "BankItem",
    "expand_bank",
    "general_preference_pairs",
    "write_jsonl",
]
