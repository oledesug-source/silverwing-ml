"""Model runtime: lifecycle management and server orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from foundation.inference import Generator, InferenceConfig


@dataclass
class RuntimeConfig:
    """Configuration for the serving runtime."""

    checkpoint_path: str = "experiments/checkpoints/sft-combined/best.pt"
    model_config_path: str = "configs/model.yaml"
    tokenizer_dir: str = "experiments/tokenizer"
    device: str = "cpu"
    host: str = "0.0.0.0"
    port: int = 8000
    max_new_tokens: int = 128

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "model_config_path": self.model_config_path,
            "tokenizer_dir": self.tokenizer_dir,
            "device": self.device,
            "host": self.host,
            "port": self.port,
            "max_new_tokens": self.max_new_tokens,
        }

    @classmethod
    def from_yaml(cls, path: str) -> RuntimeConfig:
        from pathlib import Path
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        section = raw.get("serving", raw.get("runtime", raw))
        return cls(**{k: v for k, v in section.items() if hasattr(cls, k)})


class Runtime:
    """Manages model lifecycle and HTTP serving.

    Usage::

        rt = Runtime(RuntimeConfig(checkpoint_path="experiments/checkpoints/sft-combined/best.pt"))
        rt.load()     # loads the model
        rt.serve()    # starts the HTTP server (blocking)
        rt.unload()   # releases model from memory
    """

    def __init__(self, config: RuntimeConfig, log=print) -> None:
        self._config = config
        self._log = log
        self._generator: Generator | None = None

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None

    def load(self) -> None:
        """Load the model into memory."""
        self._log(f"Loading model from {self._config.checkpoint_path}")
        inf_cfg = InferenceConfig(
            checkpoint_path=self._config.checkpoint_path,
            model_config_path=self._config.model_config_path,
            tokenizer_dir=self._config.tokenizer_dir,
            device=self._config.device,
            max_new_tokens=self._config.max_new_tokens,
        )
        self._generator = Generator.from_config(inf_cfg)
        self._log("Model loaded successfully")

    def unload(self) -> None:
        """Release the model from memory."""
        self._generator = None
        self._log("Model unloaded")

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            raise RuntimeError("Model not loaded. Call runtime.load() first.")
        return self._generator

    def health(self) -> dict[str, Any]:
        """Return health status."""
        return {
            "status": "ok" if self.is_loaded else "no_model",
            "loaded": self.is_loaded,
            "config": self._config.to_dict(),
        }

    def serve(self) -> None:
        """Start the HTTP server (blocking)."""
        import http.server
        from .api.server import SilverwingHandler

        SilverwingHandler.server_model = self.generator
        SilverwingHandler.server_config = self._config

        server = http.server.HTTPServer(
            (self._config.host, self._config.port),
            SilverwingHandler,
        )
        self._log(f"Serving on {self._config.host}:{self._config.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self._log("Shutting down...")
        finally:
            server.server_close()
