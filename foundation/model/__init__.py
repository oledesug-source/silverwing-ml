"""Silverwing Decoder V2 — modern causal transformer backbone.

RMSNorm pre-norm, RoPE, Grouped Query Attention, SwiGLU FFN, tied embeddings.
Targets ~100M parameters per configs/foundation.yaml and trains on CPU.
"""

from .config import ModelConfig
from .model import SilverwingDecoder, build_model
from .rope import apply_rope, precompute_rope_cache

__all__ = [
    "ModelConfig",
    "SilverwingDecoder",
    "apply_rope",
    "build_model",
    "precompute_rope_cache",
]
