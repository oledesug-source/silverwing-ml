"""Byte Pair Encoding (BPE) tokenizer for transformer models.

A from-scratch implementation of BPE tokenization that does not depend on
HuggingFace or any external tokenizer library.  Used by the transformer
module to convert text to token IDs and back.

Algorithm:
    1. Build vocabulary of characters from training text.
    2. Count pairs of consecutive tokens (bigrams).
    3. Merge the most frequent pair into a new token.
    4. Repeat until vocabulary reaches target size.

Reference: "Neural Machine Translation of Rare Words with Subword Units"
(Sennrich et al., 2015).
"""

from __future__ import annotations

import re
from collections import Counter


class BPETokenizer:
    """Byte Pair Encoding tokenizer.

    Args:
        vocab_size: Target vocabulary size after training.
        min_frequency: Minimum frequency for a pair to be merged.

    Attributes:
        vocab_size: The target vocabulary size.
        token_to_id: Mapping from token string → integer ID.
        id_to_token: Mapping from integer ID → token string.
        merges: Ordered list of merge pairs.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.vocab: list[str] = []
        self._word_splits: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _get_stats(
        vocab: dict[str, list[str]],
    ) -> Counter:
        """Count pair frequencies across the vocabulary.

        Args:
            vocab: Mapping from word → list of subword symbols.

        Returns:
            Counter of (symbol1, symbol2) bigram frequencies.
        """
        pairs: Counter = Counter()
        for _word, chars in vocab.items():
            if len(chars) < 2:
                continue
            for pair in zip(chars[:-1], chars[1:]):
                pairs[pair] += 1  # count each bigram occurrence
        return pairs

    @staticmethod
    def _merge_pair(
        a: str, b: str, vocab: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Merge all occurrences of pair (a, b) into 'ab' in the vocab."""
        new_vocab = {}
        for word, chars in vocab.items():
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(a + b)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            new_vocab[word] = new_chars
        return new_vocab

    def train(self, text: str) -> None:
        """Train the BPE tokenizer on the given text.

        Args:
            text: Training corpus text.
        """
        # Split into words
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        word_counts = Counter(w for w in words if w)

        # Build initial vocabulary: characters of each word split with </w> end marker
        vocab: dict[str, list[str]] = {}
        for word, _count in word_counts.items():
            chars = list(word) + ["</w>"]
            vocab[word] = chars

        # Collect all unique characters as base vocabulary
        base_chars: set[str] = set()
        for chars in vocab.values():
            base_chars.update(chars)

        # Initialize vocabulary
        self.vocab = sorted(base_chars)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = dict(enumerate(self.vocab))
        self.merges = []

        # Perform merges
        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            # Find the most frequent pair that meets min_frequency
            best_pair = None
            best_count = 0
            for pair, count in pairs.most_common():
                if count >= self.min_frequency and count > best_count:
                    best_pair = pair
                    best_count = count
                    break

            if best_pair is None:
                break

            a, b = best_pair
            self.merges.append((a, b))
            vocab = self._merge_pair(a, b, vocab)

            # Add merged token to vocabulary
            new_token = a + b
            if new_token not in self.token_to_id:
                idx = len(self.vocab)
                self.vocab.append(new_token)
                self.token_to_id[new_token] = idx
                self.id_to_token[idx] = new_token

        self._word_splits = vocab

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply BPE merges to a single word."""
        if not self.merges:
            return list(word) + ["</w>"]

        chars = list(word) + ["</w>"]
        # Iteratively apply merges
        for a, b in self.merges:
            if a + b in chars or b in chars:
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new_chars.append(a + b)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
        return chars

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids: list[int] = []
        for word in words:
            tokens = self._tokenize_word(word)
            for tok in tokens:
                # Strip end-of-word marker for lookup
                clean_tok = tok.replace("</w>", "")
                if clean_tok in self.token_to_id:
                    token_ids.append(self.token_to_id[clean_tok])
                elif tok in self.token_to_id:
                    token_ids.append(self.token_to_id[tok])
                else:
                    # Fall back to character-level encoding
                    for ch in clean_tok:
                        token_ids.append(self.token_to_id.get(ch, 0))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        tokens = [self.id_to_token.get(tid, "") for tid in token_ids]
        # Join and strip end-of-word markers
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of texts. Pads to the longest sequence."""
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(e) for e in encoded)
        pad_id = self.token_to_id.get("", 0)
        return [e + [pad_id] * (max_len - len(e)) for e in encoded]

    @property
    def actual_vocab_size(self) -> int:
        """Return the actual vocabulary size after training."""
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)
