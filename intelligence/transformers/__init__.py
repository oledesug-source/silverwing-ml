"""Transformer architecture — attention mechanisms, encoders, and decoders.

A from-scratch (no torch required for core abstractions) implementation of
the transformer architecture as described in "Attention Is All You Need"
(Vaswani et al., 2017).  Provides:

    - Scaled Dot-Product Attention (mathematical + numpy reference)
    - Multi-Head Attention (with masking support)
    - Positional Encoding (sinusoidal + learned)
    - Transformer Encoder / Decoder layers
    - Full Transformer model with configurable depth, heads, and dims
    - BPE tokenization utilities
"""

from intelligence.transformers.attention import (
    MaskedMultiHeadAttention,
    MultiHeadAttention,
    scaled_dot_product_attention,
    softmax,
)
from intelligence.transformers.bpe_tokenizer import BPETokenizer
from intelligence.transformers.model import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from intelligence.transformers.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)

__all__ = [
    "BPETokenizer",
    "LearnedPositionalEncoding",
    "MaskedMultiHeadAttention",
    "MultiHeadAttention",
    "SinusoidalPositionalEncoding",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "scaled_dot_product_attention",
    "softmax",
]
