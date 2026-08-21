"""Agent — wraps intelligence modules into a callable unit.

An ``Agent`` bundles the foundation Generator with optional intelligence
modules (planner, reasoner, memory) and the capability registry.  It
provides high-level ``chat`` and ``use_tool`` methods consumed by the
``Orchestrator``.

All intelligence modules are optional — the Agent works with or without
them (torch-dependent modules are imported lazily).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from intelligence.memory.context import WorkingMemory
from intelligence.tools.protocol import ToolResult

from .capabilities import CapabilityRegistry
from .context import RequestContext
from .sandbox import Sandbox

logger = logging.getLogger(__name__)


def _import_generator():
    """Lazy import of foundation Generator (requires torch)."""
    from foundation.inference import Generator
    return Generator


def _import_planner():
    """Lazy import of intelligence Planner (requires torch via Generator)."""
    from intelligence.planning.planner import Planner
    return Planner


def _import_reasoning_engine():
    """Lazy import of intelligence ReasoningEngine (requires torch)."""
    from intelligence.reasoning.engine import ReasoningEngine
    return ReasoningEngine


@dataclass
class Agent:
    """Wraps intelligence modules into a callable unit.

    Attributes:
        generator: The foundation Generator (optional, requires torch).
        capability_registry: Registry of available capabilities.
        sandbox: Safe execution environment.
        planner: Task decomposition planner (optional).
        reasoner: Logical reasoning engine (optional).
        memory: Shared working memory across requests.
    """

    capability_registry: CapabilityRegistry
    sandbox: Sandbox = field(default_factory=Sandbox)
    generator: Any = None  # foundation.inference.Generator | None
    planner: Any = None  # intelligence.planning.Planner | None
    reasoner: Any = None  # intelligence.reasoning.ReasoningEngine | None
    memory: WorkingMemory = field(default_factory=WorkingMemory)

    @classmethod
    def from_config(
        cls,
        capability_registry: CapabilityRegistry | None = None,
        generator: Any = None,
        max_tokens: int = 512,
    ) -> Agent:
        """Build an Agent from configuration.

        Creates a ``CapabilityRegistry`` and registers built-in tools if
        none is provided.  The *generator* is optional — when ``None`` the
        Agent can still execute tools directly but cannot generate text.
        """
        from .tools import register_builtin_tools

        if capability_registry is None:
            capability_registry = CapabilityRegistry()
            register_builtin_tools(capability_registry)

        agent = cls(
            capability_registry=capability_registry,
            generator=generator,
        )

        if generator is not None:
            Planner = _import_planner()
            ReasoningEngine = _import_reasoning_engine()
            agent.planner = Planner(generator, max_new_tokens=max_tokens)
            agent.reasoner = ReasoningEngine(generator, max_new_tokens=max_tokens)

        return agent

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def chat(self, message: str, context: RequestContext) -> str:
        """Generate a response to *message* using the Generator.

        Returns the generated text, or an error message if no Generator
        is available.
        """
        if self.generator is None:
            return "Generator not available. Use use_tool() for direct tool calls."

        # Build prompt from working memory context
        context.add_user_message()
        prompt = context.working_memory.build_context()

        result = self.generator.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
        )
        return result.text

    def use_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool directly by name with the given arguments."""
        cap = self.capability_registry.get(tool_name)
        if cap is None:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=f"Unknown capability: {tool_name}",
            )
        if cap.fn is None:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=f"Capability '{tool_name}' has no implementation",
            )
        return self.sandbox.execute(cap.fn, tool_name=tool_name, **args)

    # ------------------------------------------------------------------
    # Convenience: lazy construction from checkpoints
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config_path: str = "configs/model.yaml",
        tokenizer_dir: str = "experiments/tokenizer",
        device: str = "cpu",
        max_new_tokens: int = 256,
        capability_registry: CapabilityRegistry | None = None,
    ) -> Agent:
        """Build an Agent with a loaded Generator from a checkpoint."""
        Generator = _import_generator()
        from foundation.inference import InferenceConfig

        cfg = InferenceConfig(
            checkpoint_path=checkpoint_path,
            model_config_path=model_config_path,
            tokenizer_dir=tokenizer_dir,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        generator = Generator.from_config(cfg)
        return cls.from_config(
            capability_registry=capability_registry,
            generator=generator,
            max_tokens=max_new_tokens,
        )
