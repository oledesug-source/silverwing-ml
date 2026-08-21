"""Capability registry.

Central registry for all capabilities in the platform.  Wraps
``intelligence.tools.protocol.ToolRegistry`` internally so existing
intelligence modules continue to work unchanged.
"""

from __future__ import annotations

from intelligence.tools.protocol import (
    Tool,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)

from .schema import CapabilitySchema


class CapabilityRegistry:
    """Registry of capabilities with discovery, search, and execution.

    Wraps ``ToolRegistry`` from ``intelligence.tools.protocol`` and adds
    schema-level metadata, enable/disable, and tag-based search.

    Usage::

        registry = CapabilityRegistry()
        registry.register(CapabilitySchema(
            name="calculator",
            description="Evaluate a math expression",
            input_schema={"expression": {"type": "string"}},
            fn=lambda expression: str(eval(expression)),
            tags=["math", "safe"],
        ))
        caps = registry.search(tags=["math"])
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilitySchema] = {}
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(self._tool_registry)

    def register(self, schema: CapabilitySchema) -> None:
        """Register a capability schema."""
        self._capabilities[schema.name] = schema
        self._tool_registry.register(Tool(
            name=schema.name,
            description=schema.description,
            parameters={
                k: str(v.get("type", "any"))
                for k, v in schema.input_schema.items()
            } if schema.input_schema else {},
            fn=schema.fn,
        ))

    def unregister(self, name: str) -> bool:
        """Remove a capability. Returns True if it existed."""
        if name in self._capabilities:
            del self._capabilities[name]
            return True
        return False

    def get(self, name: str) -> CapabilitySchema | None:
        """Retrieve a capability by name."""
        return self._capabilities.get(name)

    def list(self, enabled_only: bool = True) -> list[CapabilitySchema]:
        """Return all registered capabilities."""
        caps = list(self._capabilities.values())
        if enabled_only:
            caps = [c for c in caps if c.enabled]
        return caps

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        capability_type: str | None = None,
    ) -> list[CapabilitySchema]:
        """Search capabilities by text query, tags, or type."""
        results = list(self._capabilities.values())

        if query:
            q = query.lower()
            results = [
                c for c in results
                if q in c.name.lower() or q in c.description.lower()
            ]

        if tags:
            tag_set = set(tags)
            results = [
                c for c in results
                if tag_set & set(c.tags)
            ]

        if capability_type:
            results = [c for c in results if c.capability_type == capability_type]

        return results

    def enable(self, name: str) -> bool:
        """Enable a capability. Returns True if found."""
        cap = self._capabilities.get(name)
        if cap is not None:
            cap.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a capability. Returns True if found."""
        cap = self._capabilities.get(name)
        if cap is not None:
            cap.enabled = False
            return True
        return False

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
