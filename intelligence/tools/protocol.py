"""Tool-use protocol.

Provides a structured way for the model to:

1. **Register** tools with schemas (name, description, parameters).
2. **Parse** tool calls from model output (``<tool:name>args</tool>`` format).
3. **Execute** tool calls and feed results back to the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """A registered tool with its metadata and function."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    fn: Callable[..., Any] | None = None

    def to_prompt(self) -> str:
        """Render tool description for inclusion in the system prompt."""
        params = ", ".join(
            f"{k}: {v}" for k, v in self.parameters.items()
        ) if self.parameters else "none"
        return f"- {self.name}: {self.description}\n  Parameters: {params}"


@dataclass
class ToolCall:
    """A parsed tool call from model output."""

    tool_name: str
    arguments: str
    raw_text: str = ""

    @property
    def args_dict(self) -> dict[str, Any]:
        """Try to parse arguments as key=value pairs."""
        result: dict[str, Any] = {}
        for part in self.arguments.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                result[k.strip()] = v.strip().strip("\"'")
            elif part:
                result["input"] = part
        return result


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_name: str
    output: str
    success: bool = True
    error: str = ""


TOOL_CALL_PATTERN = re.compile(
    r"<tool:(\w+)>(.*?)</tool>", re.DOTALL
)


class ToolRegistry:
    """Registry of available tools.

    Usage::

        registry = ToolRegistry()
        registry.register(Tool(
            name="calculator",
            description="Evaluate a math expression",
            parameters={"expression": "math expression string"},
            fn=lambda expression: str(eval(expression)),
        ))
        calls = registry.parse_calls(model_output)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def system_prompt(self) -> str:
        """Generate the system prompt section for available tools."""
        if not self._tools:
            return "No tools available."
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt())
        lines.append(
            "\nTo use a tool, output: <tool:name>param=value, ...</tool>"
        )
        return "\n".join(lines)

    def parse_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from model output text."""
        calls: list[ToolCall] = []
        for match in TOOL_CALL_PATTERN.finditer(text):
            calls.append(ToolCall(
                tool_name=match.group(1),
                arguments=match.group(2).strip(),
                raw_text=match.group(0),
            ))
        return calls


class ToolExecutor:
    """Execute tool calls and manage the tool-use loop.

    Usage::

        registry = ToolRegistry()
        registry.register(...)
        executor = ToolExecutor(registry)
        result = executor.execute_call(ToolCall("calculator", "2+2"))
        print(result.output)  # "4"
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self._registry.get(call.tool_name)
        if tool is None:
            return ToolResult(
                tool_name=call.tool_name,
                output="",
                success=False,
                error=f"Unknown tool: {call.tool_name}",
            )
        if tool.fn is None:
            return ToolResult(
                tool_name=call.tool_name,
                output="",
                success=False,
                error=f"Tool {call.tool_name} has no implementation",
            )
        try:
            args = call.args_dict
            output = tool.fn(**args)
            return ToolResult(
                tool_name=call.tool_name,
                output=str(output),
                success=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=call.tool_name,
                output="",
                success=False,
                error=str(exc),
            )

    def execute_all(self, text: str) -> list[ToolResult]:
        """Parse and execute all tool calls found in text."""
        calls = self._registry.parse_calls(text)
        return [self.execute_call(c) for c in calls]

    def format_results(self, results: list[ToolResult]) -> str:
        """Format tool results for inclusion in the next model prompt."""
        lines: list[str] = []
        for r in results:
            if r.success:
                lines.append(f"Tool {r.tool_name} output: {r.output}")
            else:
                lines.append(f"Tool {r.tool_name} error: {r.error}")
        return "\n".join(lines)
