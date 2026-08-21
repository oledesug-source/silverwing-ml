"""Transformer encoder and decoder layers.

Implements the core building blocks of the Transformer architecture:

    - ``TransformerEncoderLayer``: self-attention + FFN with residual connections
    - ``TransformerDecoderLayer``: masked self-attention + cross-attention + FFN
    - ``TransformerEncoder``: stacked encoder layers
    - ``TransformerDecoder``: stacked decoder layers
    - ``Transformer``: full encoder-decoder model

All implementations are numpy-based and do not require torch.
"""

from __future__ import annotations

import numpy as np

from .attention import MaskedMultiHeadAttention, MultiHeadAttention


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + x**3 / 3.0)))


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization (no learnable scale/shift in this minimal impl)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class FeedForward:
    """Position-wise feed-forward network.

    FFN(x) = max(0, x W1 + b1) W2 + b2
    Uses GELU activation by default.

    Args:
        d_model:    Input/output dimension.
        d_ff:       Hidden dimension (typically 4 * d_model).
        dropout_rate: Dropout probability for regularization.
        activation: Activation function name ("gelu" or "relu").
        seed:       Random seed for weight initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        seed: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation = activation
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (d_model + d_ff))
        self.W1 = rng.uniform(-limit, limit, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.uniform(-limit, limit, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
        self.training = True

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, x)
        return gelu(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x -> FFN -> dropout -> output."""
        out = self._activate(np.matmul(x, self.W1) + self.b1)
        # Apply dropout to hidden layer
        if self.training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            rng = np.random.default_rng()
            drop_mask = rng.random(out.shape) < keep_prob
            out = out * drop_mask / keep_prob
        out = np.matmul(out, self.W2) + self.b2
        # Apply dropout to output
        if self.training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            rng = np.random.default_rng()
            drop_mask = rng.random(out.shape) < keep_prob
            out = out * drop_mask / keep_prob
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TransformerEncoderLayer:
    """A single transformer encoder layer.

    Architecture:
        1. Multi-head self-attention (with residual + layer norm)
        2. Position-wise feed-forward (with residual + layer norm)

    Args:
        d_model:      Model dimension.
        num_heads:    Number of attention heads.
        d_ff:         Feed-forward hidden dimension.
        dropout_rate: Dropout probability.
        seed:         Random seed for initialization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout_rate = dropout_rate
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout_rate=dropout_rate, seed=seed,
        )
        self.feed_forward = FeedForward(
            d_model, self.d_ff, dropout_rate=dropout_rate, seed=seed,
        )
        self.training = True
        self.self_attn.training = self.training
        self.feed_forward.training = self.training

    def forward(
        self,
        x: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            x:        Input tensor (seq_len, d_model) or (batch, seq_len, d_model).
            src_mask: Optional additive mask for self-attention.

        Returns:
            Tuple of (output, attention_weights).
        """
        # Self-attention sublayer
        attn_out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        attn_out = layer_norm(x + attn_out)

        # Feed-forward sublayer
        ff_out = self.feed_forward(attn_out)
        ff_out = layer_norm(attn_out + ff_out)

        return ff_out, attn_weights

    def __call__(
        self,
        x: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.forward(x, src_mask)

    def set_training(self, training: bool) -> None:
        """Set training/evaluation mode for all sublayers."""
        self.training = training
        self.self_attn.training = training
        self.feed_forward.training = training


class TransformerDecoderLayer:
    """A single transformer decoder layer.

    Architecture:
        1. Masked multi-head self-attention (causal)
        2. Multi-head cross-attention (encoder -> decoder)
        3. Position-wise feed-forward
        Each sublayer has residual connections and layer normalization.

    Args:
        d_model:      Model dimension.
        num_heads:    Number of attention heads.
        d_ff:         Feed-forward hidden dimension.
        dropout_rate: Dropout probability.
        seed:         Random seed for initialization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout_rate = dropout_rate
        self.self_attn = MaskedMultiHeadAttention(
            d_model, num_heads, dropout_rate=dropout_rate, seed=seed,
        )
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout_rate=dropout_rate, seed=seed + 1 if seed else None,
        )
        self.feed_forward = FeedForward(
            d_model, self.d_ff, dropout_rate=dropout_rate, seed=seed + 2 if seed else None,
        )
        self.training = True
        self.self_attn.training = self.training
        self.cross_attn.training = self.training
        self.feed_forward.training = self.training

    def forward(
        self,
        x: np.ndarray,
        enc_output: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            x:         Target input (seq_len, d_model) or (batch, seq_len, d_model).
            enc_output: Encoder output to cross-attend to.
            src_mask:  Optional mask for encoder-decoder attention.
            tgt_mask:  Optional mask for self-attention (overridden by causal mask).

        Returns:
            Tuple of (output, self_attn_weights, cross_attn_weights).
        """
        # Masked self-attention
        self_out, self_weights = self.self_attn(x, x, x, mask=tgt_mask)
        self_out = layer_norm(x + self_out)

        # Cross-attention
        cross_out, cross_weights = self.cross_attn(self_out, enc_output, enc_output, mask=src_mask)
        cross_out = layer_norm(self_out + cross_out)

        # Feed-forward
        ff_out = self.feed_forward(cross_out)
        ff_out = layer_norm(cross_out + ff_out)

        return ff_out, self_weights, cross_weights

    def __call__(
        self,
        x: np.ndarray,
        enc_output: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.forward(x, enc_output, src_mask, tgt_mask)

    def set_training(self, training: bool) -> None:
        self.training = training
        self.self_attn.training = training
        self.cross_attn.training = training
        self.feed_forward.training = training


class TransformerEncoder:
    """Stack of ``num_layers`` ``TransformerEncoderLayer`` instances.

    Args:
        num_layers:   Number of encoder layers to stack.
        d_model:      Model dimension.
        num_heads:    Number of attention heads.
        d_ff:         Feed-forward hidden dimension.
        dropout_rate: Dropout probability.
        seed:         Base random seed (each layer gets a unique offset).
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = [
            TransformerEncoderLayer(
                d_model, num_heads, d_ff, dropout_rate,
                seed=seed + i if seed else None,
            )
            for i in range(num_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Forward pass through all layers.

        Returns:
            Tuple of (output, list_of_attention_weights_per_layer).
        """
        all_weights = []
        for layer in self.layers:
            x, weights = layer(x, src_mask=src_mask)
            all_weights.append(weights)
        return x, all_weights

    def __call__(
        self,
        x: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        return self.forward(x, src_mask)

    def set_training(self, training: bool) -> None:
        for layer in self.layers:
            layer.set_training(training)


class TransformerDecoder:
    """Stack of ``num_layers`` ``TransformerDecoderLayer`` instances.

    Args:
        num_layers:   Number of decoder layers to stack.
        d_model:      Model dimension.
        num_heads:    Number of attention heads.
        d_ff:         Feed-forward hidden dimension.
        dropout_rate: Dropout probability.
        seed:         Base random seed.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = [
            TransformerDecoderLayer(
                d_model, num_heads, d_ff, dropout_rate,
                seed=seed + i * 10 if seed else None,
            )
            for i in range(num_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        enc_output: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass through all layers.

        Returns:
            Tuple of (output, self_attn_weights_list, cross_attn_weights_list).
        """
        all_self_weights = []
        all_cross_weights = []
        for layer in self.layers:
            x, self_w, cross_w = layer(x, enc_output, src_mask, tgt_mask)
            all_self_weights.append(self_w)
            all_cross_weights.append(cross_w)
        return x, all_self_weights, all_cross_weights

    def __call__(
        self,
        x: np.ndarray,
        enc_output: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        return self.forward(x, enc_output, src_mask, tgt_mask)

    def set_training(self, training: bool) -> None:
        for layer in self.layers:
            layer.set_training(training)


class Transformer:
    """Full encoder-decoder Transformer model.

    Combines:
        - Token embedding (randomly initialized)
        - Sinusoidal positional encoding
        - Transformer encoder (stacked layers)
        - Transformer decoder (stacked layers)
        - Output projection (logits over vocabulary)

    Args:
        vocab_size:     Size of the token vocabulary.
        d_model:        Model dimension (must be divisible by ``num_heads``).
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads:      Number of attention heads.
        d_ff:           Feed-forward hidden dimension (default: 4 * d_model).
        max_seq_len:    Maximum sequence length for positional encoding.
        dropout_rate:   Dropout probability.
        seed:           Random seed for reproducible initialization.

    Example:
        >>> model = Transformer(
        ...     vocab_size=1000, d_model=128,
        ...     num_encoder_layers=2, num_decoder_layers=2, num_heads=4,
        ... )
        >>> # x_enc: (seq_len, d_model) token embeddings, x_dec: same
        >>> out, enc_w, dec_self_w, dec_cross_w = model(x_enc, x_dec)
        >>> logits = model.project(out)  # (seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        rng = np.random.default_rng(seed)

        # Token embeddings
        self.token_embedding = rng.standard_normal((vocab_size, d_model)) * 0.02

        # Positional encoding
        from .positional_encoding import SinusoidalPositionalEncoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Encoder and decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff,
            dropout_rate=dropout_rate, seed=seed,
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff,
            dropout_rate=dropout_rate, seed=seed,
        )

        # Output projection
        limit = np.sqrt(6.0 / (d_model + vocab_size))
        self.output_projection = rng.uniform(
            -limit, limit, (d_model, vocab_size)
        )

    def encode(
        self,
        token_ids: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Encode a sequence of token IDs.

        Args:
            token_ids: 1-D array of token IDs (seq_len,).

        Returns:
            Tuple of (encoder_output, encoder_attention_weights).
        """
        seq_len = len(token_ids)
        positions = np.arange(seq_len)
        embeddings = self.token_embedding[token_ids]  # (seq_len, d_model)
        embeddings = embeddings + self.pos_encoder(positions)
        return self.encoder(embeddings, src_mask=src_mask)

    def decode(
        self,
        token_ids: np.ndarray,
        enc_output: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Decode a sequence of token IDs given encoder output.

        Args:
            token_ids:   Target token IDs (seq_len,).
            enc_output:  Output from ``encode()``.

        Returns:
            Tuple of (decoder_output, self_attn_weights, cross_attn_weights).
        """
        seq_len = len(token_ids)
        positions = np.arange(seq_len)
        embeddings = self.token_embedding[token_ids]
        embeddings = embeddings + self.pos_encoder(positions)
        return self.decoder(
            embeddings, enc_output,
            src_mask=src_mask, tgt_mask=tgt_mask,
        )

    def project(self, decoder_output: np.ndarray) -> np.ndarray:
        """Project decoder output to vocabulary logits.

        Args:
            decoder_output: (seq_len, d_model) or (batch, seq_len, d_model).

        Returns:
            Logits of shape (seq_len, vocab_size).
        """
        return np.matmul(decoder_output, self.output_projection)

    def forward(
        self,
        enc_ids: np.ndarray,
        dec_ids: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list, list, list]:
        """Full forward pass: encode -> decode -> project.

        Returns:
            Tuple of (logits, enc_weights, dec_self_weights, dec_cross_weights).
        """
        enc_out, enc_w = self.encode(enc_ids, src_mask)
        dec_out, dec_s, dec_c = self.decode(
            dec_ids, enc_out, src_mask, tgt_mask,
        )
        logits = self.project(dec_out)
        return logits, enc_w, dec_s, dec_c

    def __call__(
        self,
        enc_ids: np.ndarray,
        dec_ids: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list, list, list]:
        return self.forward(enc_ids, dec_ids, src_mask, tgt_mask)

    def set_training(self, training: bool) -> None:
        self.encoder.set_training(training)
        self.decoder.set_training(training)
