"""M14: Native inference engine for Silverwing Decoder V2.

Efficient autoregressive generation with KV-cache, configurable sampling
(greedy, top-k, top-p, temperature, repetition penalty), and batched
generation.
"""

from .config import INFERENCE_VERSION, InferenceConfig
from .generator import GenerationResult, Generator

__all__ = [
    "INFERENCE_VERSION",
    "GenerationResult",
    "Generator",
    "InferenceConfig",
]
