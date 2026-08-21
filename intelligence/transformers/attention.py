"""Attention mechanisms for transformer models.

Implements:

    - ``scaled_dot_product_attention``: the core mathematical operation
    - ``MultiHeadAttention``: multi-head wrapper with trainable projections
    - ``MaskedMultiHeadAttention``: supports causal masking for decoders

All implementations are numpy-based and do not require torch.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        x: Input array.
        axis: Axis along which to apply softmax.
        temperature: Temperature scaling (higher = softer).

    Returns:
        Softmax-normalized array with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64) / temperature
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
    dropout_rate: float = 0.0,
    training: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V

    Args:
        query:  (seq_len_q, d_k) or (batch, heads, seq_len_q, d_k)
        key:    (seq_len_k, d_k) or (batch, heads, seq_len_k, d_k)
        value:  (seq_len_v, d_v) or (batch, heads, seq_len_v, d_v)
        mask:   Optional mask of shape broadcastable to (seq_len_q, seq_len_k).
                Additive mask: -inf for masked positions.
        dropout_rate: Probability of zeroing attention weights (training only).
        training: Whether in training mode (enables dropout).
        rng:    Random generator for dropout.

    Returns:
        Tuple of (output, attention_weights).
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)

    d_k = q.shape[-1]
    # Scaled dot-product: (seq_len_q, seq_len_k)
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)

    if training and dropout_rate > 0.0 and rng is not None:
        keep_prob = 1.0 - dropout_rate
        drop_mask = rng.random(weights.shape) < keep_prob
        weights = weights * drop_mask / keep_prob

    output = np.matmul(weights, v)
    return output, weights


class MultiHeadAttention:
    """Multi-head attention mechanism.

    Splits the query/key/value projections into ``num_heads`` independent
    attention heads, applies scaled dot-product attention to each, then
    concatenates and projects the results.

    Args:
        d_model:      Model dimension (must be divisible by ``num_heads``).
        num_heads:    Number of attention heads.
        d_value:      Per-head value dimension (defaults to ``d_model // num_heads``).
        dropout_rate: Attention dropout probability.
        seed:         Optional random seed for reproducible initialization.

    Attributes:
        d_model, num_heads, d_value, d_query, dropout_rate.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_value: int | None = None,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_query = d_model // num_heads
        self.d_value = d_value or self.d_query
        self.dropout_rate = dropout_rate
        self.training = True
        self._rng = np.random.default_rng(seed)

        # Initialize projection weights (Xavier/Glorot uniform)
        limit = np.sqrt(6.0 / (d_model + self.d_query))
        self.W_q = self._rng.uniform(-limit, limit, (d_model, d_model))
        self.W_k = self._rng.uniform(-limit, limit, (d_model, d_model))
        self.W_v = self._rng.uniform(-limit, limit, (d_model, d_model))
        self.W_o = self._rng.uniform(-limit, limit, (d_model, d_model))

        # Bias terms
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, d_query) and transpose.

        Args:
            x: Array of shape (batch, seq_len, d_model) or (seq_len, d_model).

        Returns:
            Array of shape (batch, num_heads, seq_len, d_query) or
            (num_heads, seq_len, d_query) for unbatched input.
        """
        if x.ndim == 2:
            seq_len = x.shape[0]
            x = x.reshape(seq_len, self.num_heads, self.d_query)
            return x.transpose(1, 0, 2)  # (heads, seq_len, d_query)
        batch, seq_len = x.shape[:2]
        x = x.reshape(batch, seq_len, self.num_heads, self.d_query)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_query)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Inverse of ``_split_heads``."""
        if x.ndim == 3:
            x = x.transpose(1, 0, 2)  # (seq_len, heads, d_query)
            return x.reshape(x.shape[0], self.d_model)
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_query)
        return x.reshape(x.shape[0], x.shape[1], self.d_model)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass of multi-head attention.

        Args:
            query:  (seq_len, d_model) or (batch, seq_len, d_model)
            key:    Same shape as query or (kv_len, d_model)
            value:  Same shape as query or (kv_len, d_model)
            mask:   Optional additive mask broadcastable to (seq_len, kv_len).

        Returns:
            Tuple of (output, attention_weights):
                - output: (seq_len, d_model) or (batch, seq_len, d_model)
                - attention_weights: concatenated across heads, shape
                  (num_heads, seq_len, kv_len) or (batch, num_heads, seq_len, kv_len)
        """
        # Linear projections
        q = np.matmul(query, self.W_q) + self.b_q
        k = np.matmul(key, self.W_k) + self.b_k
        v = np.matmul(value, self.W_v) + self.b_v

        # Split into heads
        q_split = self._split_heads(q)
        k_split = self._split_heads(k)
        v_split = self._split_heads(v)

        # Apply attention per head
        if q_split.ndim == 3:
            # Unbatched: (heads, seq_q, d_query)
            outputs = []
            all_weights = []
            for h in range(self.num_heads):
                out, w = scaled_dot_product_attention(
                    q_split[h], k_split[h], v_split[h],
                    mask=mask, dropout_rate=self.dropout_rate,
                    training=self.training, rng=self._rng,
                )
                outputs.append(out)
                all_weights.append(w)
            output = np.stack(outputs, axis=0)  # (heads, seq_q, d_query)
            weights = np.stack(all_weights, axis=0)  # (heads, seq_q, kv_len)
        else:
            # Batched: (batch, heads, seq_q, d_value)
            outputs = []
            all_weights = []
            for h in range(self.num_heads):
                out, w = scaled_dot_product_attention(
                    q_split[:, h], k_split[:, h], v_split[:, h],
                    mask=mask, dropout_rate=self.dropout_rate,
                    training=self.training, rng=self._rng,
                )
                outputs.append(out)
                all_weights.append(w)
            output = np.stack(outputs, axis=1)  # (batch, heads, seq_q, d_value)
            weights = np.stack(all_weights, axis=1)  # (batch, heads, seq_q, kv_len)

        # Combine heads and project
        combined = self._combine_heads(output)
        output = np.matmul(combined, self.W_o) + self.b_o

        # Apply dropout to output
        if self.training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            drop_mask = self._rng.random(output.shape) < keep_prob
            output = output * drop_mask / keep_prob

        return output, weights

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alias for ``forward``."""
        return self.forward(query, key, value, mask)


class MaskedMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with causal (left-to-right) masking.

    Used in decoder self-attention to prevent attending to future positions.
    """

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply causal masking in addition to any provided mask."""
        seq_len_q = query.shape[-2] if query.ndim >= 2 else query.shape[0]
        seq_len_k = key.shape[-2] if key.ndim >= 2 else key.shape[0]

        # Create causal mask: -inf above the diagonal
        causal_mask = np.triu(
            np.full((seq_len_q, seq_len_k), -1e9, dtype=np.float64), k=1
        )

        if mask is not None:
            combined_mask = mask + causal_mask
        else:
            combined_mask = causal_mask

        return super().forward(query, key, value, mask=combined_mask)
