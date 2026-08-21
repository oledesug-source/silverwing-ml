"""Positional encoding strategies for transformer models.

Implements both sinusoidal (fixed) and learned positional encodings.
All implementations are numpy-based and do not require torch at import time.
"""

from __future__ import annotations

import numpy as np


class SinusoidalPositionalEncoding:
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Pre-computes a lookup table of shape (max_len, d_model) where:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This encoding uses relative position relationships and allows the model
    to extrapolate to sequence lengths not seen during training.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.d_model = d_model
        self.max_len = max_len
        self._encoding = self._build_encoding()

    def _build_encoding(self) -> np.ndarray:
        """Build the sinusoidal encoding table."""
        position = np.arange(self.max_len)[:, np.newaxis]  # (max_len, 1)
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)
        )  # (d_model/2,)
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Return positional encodings for given positions.

        Args:
            positions: 1-D array of integer positions (0-based).

        Returns:
            Array of shape (len(positions), d_model).
        """
        positions = np.asarray(positions)
        if positions.max() >= self.max_len:
            raise ValueError(
                f"Position {positions.max()} exceeds max_len ({self.max_len}). "
                f"Increase max_len when creating the encoding."
            )
        return self._encoding[positions]


class LearnedPositionalEncoding:
    """Learned positional encoding.

    Unlike sinusoidal encoding, learned positions are random-initialized
    and trained alongside the model.  Supports arbitrary vocabulary sizes.
    """

    def __init__(self, d_model: int, max_len: int = 512, seed: int | None = None) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.d_model = d_model
        self.max_len = max_len
        rng = np.random.default_rng(seed)
        self._encoding = rng.standard_normal((max_len, d_model)) * 0.02

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Return learned positional encodings for given positions."""
        positions = np.asarray(positions)
        if positions.max() >= self.max_len:
            raise ValueError(
                f"Position {positions.max()} exceeds max_len ({self.max_len})."
            )
        return self._encoding[positions]

    def get_all(self) -> np.ndarray:
        """Return the full encoding table (max_len, d_model)."""
        return self._encoding
