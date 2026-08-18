"""Authoritative architecture config for Silverwing Decoder V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

MODEL_VERSION = "silverwing-decoder-v2"


def _default_dict() -> dict[str, Any]:
    return {
        f.name: f.default for f in fields(ModelConfig) if not f.name.startswith("_")
    }


_COERCERS = {"int": int, "float": float, "bool": bool}


def _coerce(value: Any, annotation: str) -> Any:
    coercer = _COERCERS.get(annotation)
    if coercer is None or isinstance(value, coercer):
        return value
    if annotation == "bool":
        return str(value).strip().lower() in ("true", "1", "yes", "on")
    return coercer(value)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = MODEL_VERSION
    vocab_size: int = 16384
    block_size: int = 2048
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    mlp_hidden_size: int = 2560
    mlp_activation: str = "swiglu"
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    bias: bool = False
    init_std: float = 0.02
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd {self.n_embd} must be divisible by n_head {self.n_head}"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head {self.n_head} must be divisible by n_kv_head {self.n_kv_head}"
            )
        if self.mlp_activation not in ("gelu", "swiglu"):
            raise ValueError(
                f"mlp_activation must be 'gelu' or 'swiglu', got {self.mlp_activation}"
            )
        if self.vocab_size <= 0 or self.block_size <= 0:
            raise ValueError("vocab_size and block_size must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def rope_head_dim(self) -> int:
        return self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        defaults = _default_dict()
        filtered: dict[str, Any] = {}
        for field in fields(cls):
            key = field.name
            if key in values and values[key] is not None:
                filtered[key] = _coerce(values[key], field.type)
            else:
                filtered[key] = defaults[key]
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("model", raw)
        return cls.from_dict(section)

    def expected_parameter_count(self) -> int:
        """Closed-form parameter estimate (tied embeddings counted once)."""
        embedding = (
            self.vocab_size * self.n_embd
            if self.tie_embeddings
            else self.vocab_size * self.n_embd * 2
        )
        head_dim = self.head_dim
        qkv = self.n_embd * (self.n_embd + 2 * self.n_kv_head * head_dim)
        attn = qkv + self.n_embd * self.n_embd
        if self.mlp_activation == "swiglu":
            mlp = 3 * self.n_embd * self.mlp_hidden_size
        else:
            mlp = 2 * self.n_embd * self.mlp_hidden_size
        block = attn + mlp + 2 * self.n_embd
        return embedding + self.n_layer * block + self.n_embd
