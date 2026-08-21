"""Model runtime: lifecycle management and server orchestration."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _builtin_calculator(expression: str) -> str:
    """Safe AST-based math evaluator."""
    _BINOPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval(tree)
    return str(int(result)) if isinstance(result, float) and result == int(result) else str(result)


def _builtin_read_file(path: str, max_bytes: int = 65536) -> str:
    """Read-only, size-limited file reader."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    if p.stat().st_size > max_bytes:
        raise ValueError(f"File too large: {p.stat().st_size} bytes (limit: {max_bytes})")
    return p.read_text(encoding="utf-8", errors="replace")


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

    When ``intelligence_enabled=True``, the Runtime creates an
    ``Orchestrator`` and mounts ``IntelligenceHandler`` which adds
    ``POST /chat``, ``POST /tool/execute``, and ``GET /capabilities``
    endpoints.  The legacy endpoints are always preserved.

    Usage::

        rt = Runtime(RuntimeConfig())
        rt.load()
        rt.serve()
        rt.unload()
    """

    def __init__(
        self,
        config: RuntimeConfig,
        log=print,
        intelligence_enabled: bool = False,
    ) -> None:
        self._config = config
        self._log = log
        self._generator = None
        self._intelligence_enabled = intelligence_enabled
        self._orchestrator = None
        self._capability_registry = None

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None

    def load(self) -> None:
        """Load the model into memory."""
        from foundation.inference import Generator, InferenceConfig

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

        if self._intelligence_enabled:
            self._setup_intelligence()

    def _setup_intelligence(self) -> None:
        """Create Orchestrator and CapabilityRegistry for intelligence mode."""
        from sw_platform.capabilities.registry import CapabilityRegistry
        from sw_platform.capabilities.schema import CapabilitySchema
        from sw_platform.orchestration.orchestrator import Orchestrator
        from sw_platform.permissions.policy import PermissionPolicy

        self._capability_registry = CapabilityRegistry()

        builtin_tools = [
            CapabilitySchema(
                name="calculator",
                description="Evaluate a math expression (+, -, *, /, //, %, **, parentheses)",
                input_schema={"expression": {"type": "string"}},
                tags=["math", "safe"],
                fn=_builtin_calculator,
            ),
            CapabilitySchema(
                name="read_file",
                description="Read a local file (read-only, max 64KB)",
                input_schema={"path": {"type": "string"}},
                tags=["file", "safe"],
                fn=_builtin_read_file,
            ),
        ]
        for cap in builtin_tools:
            self._capability_registry.register(cap)

        self._orchestrator = Orchestrator(
            registry=self._capability_registry,
            generator=self._generator,
            permissions=PermissionPolicy(),
        )
        self._log("Intelligence runtime initialized")

    def unload(self) -> None:
        """Release the model from memory."""
        self._generator = None
        self._orchestrator = None
        self._capability_registry = None
        self._log("Model unloaded")

    @property
    def generator(self):
        if self._generator is None:
            raise RuntimeError("Model not loaded. Call runtime.load() first.")
        return self._generator

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            raise RuntimeError(
                "Intelligence runtime not initialized. "
                "Use Runtime(intelligence_enabled=True) and call load()."
            )
        return self._orchestrator

    def health(self) -> dict[str, Any]:
        """Return health status."""
        return {
            "status": "ok" if self.is_loaded else "no_model",
            "loaded": self.is_loaded,
            "intelligence": self._intelligence_enabled,
            "config": self._config.to_dict(),
        }

    def serve(self) -> None:
        """Start the HTTP server (blocking)."""
        import http.server

        if self._intelligence_enabled and self._orchestrator is not None:
            from sw_platform.api import PlatformHandler

            PlatformHandler.server_model = self.generator
            PlatformHandler.server_config = self._config
            PlatformHandler.server_orchestrator = self._orchestrator
            PlatformHandler.server_registry = self._capability_registry

            handler_cls = PlatformHandler
            self._log(
                f"Serving (intelligence mode) on "
                f"{self._config.host}:{self._config.port}"
            )
        else:
            from ..api.server import SilverwingHandler

            SilverwingHandler.server_model = self.generator
            SilverwingHandler.server_config = self._config

            handler_cls = SilverwingHandler
            self._log(
                f"Serving on {self._config.host}:{self._config.port}"
            )

        server = http.server.HTTPServer(
            (self._config.host, self._config.port),
            handler_cls,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self._log("Shutting down...")
        finally:
            server.server_close()
