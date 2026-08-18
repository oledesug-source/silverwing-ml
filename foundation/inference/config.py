"""Configuration for M14: native Silverwing inference with KV cache decoding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

INFERENCE_VERSION = "inference-v1"


@dataclass(frozen=True)
class InferenceConfig:
    version: str = INFERENCE_VERSION
    checkpoint_path: str = "experiments/checkpoints/best.pt"
    model_config_path: str = "configs/model.yaml"
    tokenizer_dir: str = "experiments/tokenizer"
    device: str = "cpu"

    #: Decoding parameters
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    #: Generation control
    prompt_template: str | None = None
    stop_on_eos: bool = True

    #: Batching
    batch_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> InferenceConfig:
        defaults = {f.name: f.default for f in fields(cls)}
        filtered: dict[str, Any] = {}
        for f in fields(cls):
            key = f.name
            if key in values and values[key] is not None:
                filtered[key] = values[key]
            else:
                filtered[key] = defaults[key]
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("inference", raw)
        return cls.from_dict(section)
