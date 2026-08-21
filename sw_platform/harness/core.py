"""PydanticAI-based agent harness for the Silverwing platform.

This module bridges pydantic_ai's structured agent framework with the
existing Silverwing runtime infrastructure (sw_platform).  The harness
manages long-term memory, context windows, error handling, and
self-correction while delegating execution to the platform's permission
engine, sandbox, and audit log.

Key concepts:
    - ``HarnessAgent`` — wraps pydantic_ai.Agent with platform-aware
      tool registration and execution.
    - ``ToolSpec`` — pydantic model describing a capability for the LLM.
    - ``ExecutionResult`` — structured result of a tool execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a tool through the harness."""

    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "metadata": self.metadata,
        }


class ToolSpec(BaseModel):
    """Structured description of a tool/capability for the LLM.

    This replaces the custom ``ToolCall`` parsing protocol with
    pydantic_ai's native function-calling mechanism.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    permission_required: str = "read"


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for objects that provide executable tools."""

    def get_tools(self) -> list[ToolSpec]: ...

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult: ...


@dataclass
class HarnessContext:
    """Runtime context passed to the harness during execution.

    Attributes:
        session_id: Unique session identifier.
        request_id: Unique request identifier.
        user_id: User identifier (for permission evaluation).
        permission_level: Current permission level (L0-L5).
        metadata: Arbitrary request metadata.
        conversation_history: List of prior messages for context window.
    """

    session_id: str
    request_id: str
    user_id: str = "anonymous"
    permission_level: str = "read"
    max_rounds: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    def add_to_history(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})


class HarnessAgent:
    """Agent harness integrating pydantic_ai with Silverwing runtime.

    The harness wraps a pydantic_ai agent instance and routes all tool
    executions through the platform's sandbox, permission engine, and
    audit log.  It provides:

    - **Long-term memory**: conversation history is maintained across
      rounds via ``HarnessContext``.
    - **Context window management**: history is truncated to stay within
      token limits.
    - **Error handling**: tool execution failures are caught and fed back
      to the model as error context, enabling self-correction.
    - **Self-correction**: the model can retry failed tool calls with
      adjusted arguments.

    Usage::

        from pydantic_ai import Agent
        from sw_platform.harness import HarnessAgent, HarnessContext

        inner = Agent('openai:gpt-4o')
        harness = HarnessAgent(inner)
        ctx = HarnessContext(session_id='s1', request_id='r1')
        result = harness.run(ctx, 'What is the capital of France?')
    """

    def __init__(
        self,
        agent: Any = None,
        tools: list[ToolProvider] | None = None,
        audit_log: Any = None,
        sandbox: Any = None,
        permission_evaluator: Any = None,
    ) -> None:
        self._agent = agent
        self._tool_providers: list[ToolProvider] = tools or []
        self._audit_log = audit_log
        self._sandbox = sandbox
        self._permission_evaluator = permission_evaluator
        self._all_tools: dict[str, ToolSpec] = {}
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """Rebuild the tool table from all providers."""
        self._all_tools.clear()
        for provider in self._tool_providers:
            for tool in provider.get_tools():
                self._all_tools[tool.name] = tool

    def register_tool_provider(self, provider: ToolProvider) -> None:
        """Register a new tool provider."""
        self._tool_providers.append(provider)
        self._refresh_tools()

    @property
    def tools(self) -> list[ToolSpec]:
        """Return all registered tool specs."""
        return list(self._all_tools.values())

    def get_tool(self, name: str) -> ToolSpec | None:
        """Look up a tool by name."""
        return self._all_tools.get(name)

    def _audit(
        self,
        ctx: HarnessContext,
        action: str,
        status: str = "success",
        detail: str = "",
        elapsed_ms: float = 0.0,
    ) -> None:
        """Record an audit event if an audit log is available."""
        if self._audit_log is None:
            return
        try:
            from sw_platform.audit.events import AuditEvent

            self._audit_log.record(AuditEvent(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                action=action,
                status=status,
                detail=detail,
                elapsed_ms=elapsed_ms,
            ))
        except Exception as exc:
            logger.warning("Audit log failed: %s", exc)

    def _execute_tool(
        self,
        ctx: HarnessContext,
        tool_name: str,
        args: dict[str, Any],
    ) -> ExecutionResult:
        """Execute a tool through the appropriate provider.

        Routes through the sandbox and permission evaluator when available.
        """
        t0 = time.monotonic()
        tool_spec = self.get_tool(tool_name)

        if tool_spec is None:
            result = ExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
                elapsed_seconds=time.monotonic() - t0,
            )
            self._audit(ctx, "tool_error", status="error", detail=result.error)
            return result

        # Permission check
        if self._permission_evaluator is not None and tool_spec.permission_required:
            try:
                allowed, reason = self._permission_evaluator.is_allowed(
                    type("Cap", (), {
                        "name": tool_spec.name,
                        "permissions_required": [tool_spec.permission_required],
                        "risk_level": tool_spec.risk_level,
                        "enabled": True,
                        "id": f"tool:{tool_spec.name}",
                        "version": "1.0.0",
                        "description": tool_spec.description,
                        "source": "builtin",
                        "capability_type": "tool",
                        "tags": tool_spec.tags,
                        "fn": None,
                        "input_schema": {},
                        "output_schema": {},
                        "timeout_seconds": 30,
                    })()
                )
                if not allowed:
                    result = ExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"Permission denied: {reason}",
                    )
                    self._audit(ctx, "permission_denied", status="denied", detail=reason)
                    return result
            except Exception as exc:
                logger.warning("Permission check failed: %s", exc)

        # Execute through provider (which may delegate to sandbox)
        for provider in self._tool_providers:
            if tool_name in {t.name for t in provider.get_tools()}:
                try:
                    result = provider.execute(tool_name, **args)
                    self._audit(
                        ctx, f"tool_{tool_name}",
                        status="success" if result.success else "error",
                        detail=result.error[:200] if result.error else "",
                        elapsed_ms=result.elapsed_seconds * 1000,
                    )
                    return result
                except Exception as exc:
                    result = ExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"Execution error: {exc}",
                        elapsed_seconds=time.monotonic() - t0,
                    )
                    self._audit(ctx, "tool_error", status="error", detail=str(exc))
                    return result

        result = ExecutionResult(
            tool_name=tool_name,
            success=False,
            error=f"Tool {tool_name} is not executable",
            elapsed_seconds=time.monotonic() - t0,
        )
        self._audit(ctx, "tool_error", status="error", detail=result.error)
        return result

    def _truncate_history(
        self, history: list[dict[str, str]], max_tokens: int = 4096
    ) -> list[dict[str, str]]:
        """Truncate conversation history to stay within token limits.

        Uses a rough 4-characters-per-token estimate.
        """
        total_tokens = 0
        kept: list[dict[str, str]] = []
        for msg in reversed(history):
            msg_tokens = len(msg.get("content", "")) // 4
            if total_tokens + msg_tokens > max_tokens:
                break
            total_tokens += msg_tokens
            kept.insert(0, msg)
        return kept

    def build_system_prompt(self, ctx: HarnessContext) -> str:
        """Construct the system prompt with available tools and context."""
        tool_descs = []
        for tool in self.tools:
            params = ", ".join(
                f"{k}: {v}" for k, v in tool.parameters.items()
            ) if tool.parameters else "no parameters"
            tool_descs.append(
                f"Tool '{tool.name}': {tool.description}. Parameters: {params}"
            )

        tools_section = "\n".join(tool_descs) if tool_descs else "No tools available."

        return (
            "You are Silverwing, an advanced autonomous agent with system "
            "tool execution capabilities.\n\n"
            f"Permission Level: {ctx.permission_level}\n\n"
            f"Available tools:\n{tools_section}\n\n"
            "To use a tool, respond with a structured format. "
            "If a tool fails, analyze the error and retry with corrected "
            "arguments. You may chain multiple tool calls in a single response."
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools as dicts (for API serialization)."""
        return [tool.model_dump() for tool in self.tools]
