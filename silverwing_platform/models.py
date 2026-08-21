"""Model abstraction — stable inference interface.

The platform never depends on transformer internals directly.  Every model
is accessed through a :class:`ModelProvider` that accepts an
:class:`InferenceRequest` (prompt + :class:`GenerationConfig`) and returns an
:class:`InferenceResponse`.

Providers:
    - :class:`GeneratorProvider` — adapter around ``foundation.inference.Generator``
      (the native Silverwing Decoder V2).  Torch is imported lazily so the
      platform boots without it.
    - :class:`MockProvider` — deterministic fake for tests / no-model fallback.

Selection is configuration driven: a :class:`ModelRegistry`-style lookup maps a
model id to a provider class.  Multiple providers (v1, v2, specialised, fallback)
can coexist behind the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationConfig",
    "InferenceRequest",
    "InferenceResponse",
    "ModelMetadata",
    "ModelProvider",
    "GeneratorProvider",
    "MockProvider",
    "ModelProviderError",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Sampling/decoding parameters for a generation call."""

    max_new_tokens: int = 128
    min_new_tokens: int = 0
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_on_eos: bool = True
    stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "stop_on_eos": self.stop_on_eos,
            "stream": self.stream,
        }


@dataclass
class ModelMetadata:
    """Static metadata describing a model provider / version."""

    model_id: str
    version: str = "1.0.0"
    model_type: str = "decoder"
    status: str = "experimental"
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """A single inference request."""

    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)
    model_id: str = ""


@dataclass
class InferenceResponse:
    """A single inference response."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    model_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.metadata.get("prompt_tokens", 0),
            "generated_tokens": len(self.token_ids),
            "total_tokens": self.metadata.get("prompt_tokens", 0) + len(self.token_ids),
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelProviderError(Exception):
    """Raised when a model provider fails to load or infer."""


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class ModelProvider(ABC):
    """Abstract model provider interface."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Static metadata for this provider."""

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a response for *request*."""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class GeneratorProvider(ModelProvider):
    """Adapter around ``foundation.inference.Generator``.

    Torch and the Silverwing model are imported lazily so that the platform
    can be constructed (and tested) without a GPU or model checkpoint.
    """

    def __init__(self, model_id: str = "silverwing-v2") -> None:
        self._model_id = model_id
        self._generator: Any = None
        self._metadata = ModelMetadata(
            model_id=model_id,
            version="2.0.0",
            model_type="decoder",
            status="loaded",
            description="SilverWing native Decoder V2",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def generator(self) -> Any:
        """Lazily load and return the underlying foundation Generator."""
        if self._generator is None:
            self._generator = self._load()
        return self._generator

    def _load(self) -> Any:
        try:
            from foundation.inference import Generator
        except ImportError as exc:
            raise ModelProviderError(
                "foundation.inference.Generator unavailable"
            ) from exc

        gen = Generator.from_config(_default_inference_config())
        self._metadata.status = "loaded"
        return gen

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if self._generator is None:
            self._generator = self._load()
        result = self._generator.generate(
            request.prompt,
            max_new_tokens=request.config.max_new_tokens,
            temperature=request.config.temperature,
            top_k=request.config.top_k,
            top_p=request.config.top_p,
        )
        return InferenceResponse(
            text=getattr(result, "text", ""),
            token_ids=getattr(result, "token_ids", []),
            model_id=self._model_id,
            metadata={
                "prompt_tokens": len(self._generator._tokenizer.encode(request.prompt))
                if hasattr(self._generator, "_tokenizer")
                else 0,
            },
        )


def _default_inference_config() -> Any:
    """Build a default InferenceConfig from the standard config file."""
    from pathlib import Path

    cfg_path = Path("configs/inference.yaml")
    if cfg_path.exists():
        from foundation.inference import InferenceConfig

        return InferenceConfig.from_yaml(cfg_path)
    from foundation.inference import InferenceConfig

    return InferenceConfig()


class MockProvider(ModelProvider):
    """Deterministic fake provider for testing and no-model fallback.

    Returns the text ``"mock: <prompt>"`` for every request unless a custom
    response/sequence is supplied.
    """

    def __init__(
        self,
        model_id: str = "mock",
        responses: list[str] | str | None = None,
    ) -> None:
        self._model_id = model_id
        self._responses: list[str] = (
            [responses] if isinstance(responses, str) else list(responses or [])
        )
        self._index = 0
        self._metadata = ModelMetadata(
            model_id=model_id,
            version="0.1.0",
            model_type="mock",
            status="loaded",
            description="Deterministic mock provider",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if not self._responses:
            text = f"mock: {request.prompt}"
        else:
            text = self._responses[min(self._index, len(self._responses) - 1)]
            self._index += 1
        return InferenceResponse(
            text=text,
            token_ids=[],
            model_id=self._model_id,
        )
