"""Capability registry and discovery.

Promotes ``intelligence.tools.protocol.ToolRegistry`` into the canonical
capability protocol for the Intelligence Runtime.  The existing
``ToolRegistry`` is wrapped (adapter pattern) so all existing intelligence
modules continue to work unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from intelligence.tools.protocol import (
    Tool,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


@dataclass
class Capability:
    """A registered capability — extends Tool with discovery metadata."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    fn: Callable[..., Any] | None = None
    source: str = "builtin"
    tags: list[str] = field(default_factory=list)
    requires_permission: bool = False
    timeout_seconds: float = 30.0

    def to_tool(self) -> Tool:
        """Convert to a ``Tool`` for use with existing intelligence modules."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            fn=self.fn,
        )


class CapabilityRegistry:
    """Registry of capabilities with discovery support.

    Wraps ``ToolRegistry`` from ``intelligence.tools.protocol`` and adds
    discovery by tag, permission metadata, timeout configuration, and
    source tracking.

    Usage::

        registry = CapabilityRegistry()
        registry.register(Capability(
            name="calculator",
            description="Evaluate a math expression",
            parameters={"expression": "math expression string"},
            fn=lambda expression: str(eval(expression)),
            tags=["math", "safe"],
        ))
        caps = registry.discover(tags=["math"])
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(self._tool_registry)

    def register(self, capability: Capability) -> None:
        """Register a capability."""
        self._capabilities[capability.name] = capability
        self._tool_registry.register(capability.to_tool())

    def get(self, name: str) -> Capability | None:
        """Retrieve a capability by name."""
        return self._capabilities.get(name)

    def discover(self, tags: list[str] | None = None) -> list[Capability]:
        """Discover capabilities, optionally filtered by tags.

        If *tags* is ``None`` or empty, returns all capabilities.
        Otherwise returns capabilities that have **any** of the requested tags.
        """
        if not tags:
            return list(self._capabilities.values())
        tag_set = set(tags)
        return [
            cap for cap in self._capabilities.values()
            if tag_set & set(cap.tags)
        ]

    def list_capabilities(self) -> list[Capability]:
        """Return all registered capabilities."""
        return list(self._capabilities.values())

    def system_prompt(self) -> str:
        """Generate the system prompt section for available capabilities."""
        return self._tool_registry.system_prompt()

    def parse_calls(self, text: str) -> list[ToolCall]:
        """Parse capability calls from model output text."""
        return self._tool_registry.parse_calls(text)

    def execute_call(self, call: ToolCall) -> ToolResult:
        """Execute a single capability call."""
        return self._tool_executor.execute_call(call)

    def execute_all(self, text: str) -> list[ToolResult]:
        """Parse and execute all capability calls found in text."""
        return self._tool_executor.execute_all(text)

    def format_results(self, results: list[ToolResult]) -> str:
        """Format tool results for inclusion in the next model prompt."""
        return self._tool_executor.format_results(results)

    def to_tool_registry(self) -> ToolRegistry:
        """Return the underlying ``ToolRegistry`` for backward compatibility."""
        return self._tool_registry
