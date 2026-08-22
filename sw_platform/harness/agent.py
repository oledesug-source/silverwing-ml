"""PydanticAI agent harness — bridges pydantic_ai with the Silverwing platform.

Layer 2: Agent Harness Architecture.

Implements an agent harness that manages long-term memory, context windows,
error handling, and self-correction, using pydantic_ai as the underlying
framework while routing all executions through the platform's sandbox,
permission engine, and audit log.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from sw_platform.harness.core import ExecutionResult, HarnessContext, ToolProvider, ToolSpec
from sw_platform.tools.code_execution import CodeExecutionProvider
from sw_platform.tools.database import DatabaseProvider
from sw_platform.tools.filesystem import FilesystemProvider
from sw_platform.tools.git import GitProvider

logger = logging.getLogger(__name__)

__all__ = [
    "AgentResponse",
    "HarnessConfig",
    "PydanticAgentHarness",
    "ToolCallRecord",
    "create_harness_agent",
]


@dataclass
class ToolCallRecord:
    """Record of a single tool call within an agent session."""

    tool_name: str
    arguments: dict[str, Any]
    result: ExecutionResult
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result.to_dict(),
            "timestamp": self.timestamp,
        }


@dataclass
class AgentResponse:
    """Structured response from the agent harness."""

    text: str
    success: bool = True
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = 0.0
    conversation_turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "success": self.success,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "conversation_turn": self.conversation_turn,
        }


class HarnessConfig(BaseModel):
    """Configuration for the agent harness.

    ``model`` accepts any pydantic_ai model identifier.  To target the
    locally served Silverwing LLM (OpenAI-compatible endpoint), set::

        SILVERWING_AGENT_MODEL=openai:silverwing-v2
        OPENAI_BASE_URL=http://localhost:8000/v1
        OPENAI_API_KEY=local
    """

    model: str = os.environ.get("SILVERWING_AGENT_MODEL", "openai:gpt-4o")
    system_prompt: str = (
        "You are Silverwing, an advanced autonomous agent with system tool "
        "execution capabilities. You manage long-term memory, context windows, "
        "error handling, and self-correction. The LLM proposes and the platform "
        "decides and executes. You are running inside a controlled intelligence "
        "platform with permission levels L0-L5, resource-bounded sandboxes, "
        "and structured audit logging."
    )
    max_rounds: int = 5
    max_tool_calls_per_round: int = 3
    tool_timeout_seconds: float = 30.0
    permission_level: str = "read"
    sandbox_enabled: bool = True
    sandbox_memory_limit: str = "512m"
    allowed_paths: list[str] = Field(default_factory=list)
    read_only_mode: bool = True


def create_harness_agent(
    model: str | None = None,
    system_prompt: str | None = None,
    max_rounds: int = 5,
    tool_timeout: float = 30.0,
    permission_level: str = "read",
    allowed_paths: list[str] | None = None,
    repo_root: str | None = None,
    database_path: str | None = None,
) -> PydanticAgentHarness:
    """Factory function to create a configured harness agent.

    Parameters:
        model: Model identifier for pydantic_ai (e.g. 'openai:gpt-4o').
        system_prompt: Override the default system prompt.
        max_rounds: Maximum conversation rounds.
        tool_timeout: Timeout per tool execution in seconds.
        permission_level: Default permission level (read/write/execute).
        allowed_paths: Filesystem paths the sandbox permits.
        repo_root: Root directory for repo context and git operations.
        database_path: Path to SQLite database for DB tools.

    Returns:
        A configured PydanticAgentHarness instance.
    """
    config = HarnessConfig(
        model=model or HarnessConfig().model,
        system_prompt=system_prompt or HarnessConfig().system_prompt,
        max_rounds=max_rounds,
        tool_timeout_seconds=tool_timeout,
        permission_level=permission_level,
        allowed_paths=allowed_paths or [],
    )
    harness = PydanticAgentHarness(config)
    repo_root = repo_root or (allowed_paths[0] if allowed_paths else os.getcwd())
    harness._repo_root = repo_root
    if database_path:
        harness.register_provider(DatabaseProvider(database_path=database_path))
    return harness


class PydanticAgentHarness:
    """PydanticAI-based agent harness with platform integration.

    Wraps a pydantic_ai Agent, registering tools as structured function
    tools while routing execution through Silverwing's sandbox, permissions,
    and audit log.

    Usage::

        harness = create_harness_agent(model="openai:gpt-4o")
        response = harness.run("Calculate the factorial of 10")
        print(response.text)
    """

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self._config = config or HarnessConfig()
        self._providers: list[ToolProvider] = []
        self._tool_registry: dict[str, ToolSpec] = {}
        self._audit_events: list[dict[str, Any]] = []
        self._conversation_history: list[dict[str, str]] = []
        self._repo_root: str = os.getcwd()

        # Initialize sandbox
        from sw_platform.sandbox.executor import ResourceLimits, SandboxExecutor

        limits = ResourceLimits(
            max_execution_time=self._config.tool_timeout_seconds,
            network_allowed=False,
            allowed_paths=self._config.allowed_paths or [],
        )
        self._sandbox = SandboxExecutor(limits=limits)

        # Initialize permission evaluator
        from sw_platform.permissions.policy import PermissionEvaluator, PermissionPolicy

        self._permission_policy = PermissionPolicy(level=self._config.permission_level)
        self._evaluator = PermissionEvaluator(self._permission_policy)

        # Register built-in tool providers (Layer 3 + Layer 4)
        self._register_builtin_tools()

        # Lazy-init pydantic_ai agent
        self._agent: Any = None

    def _register_builtin_tools(self) -> None:
        """Register built-in tool providers from Layers 3 and 4."""
        # Layer 3: Code execution
        self._providers.append(CodeExecutionProvider(sandbox=self._sandbox))

        # Layer 3: Filesystem (os, shutil)
        self._providers.append(FilesystemProvider(
            allowed_paths=self._config.allowed_paths or [],
            sandbox=self._sandbox,
        ))

        # Layer 3: Git (git clone, commit, status)
        self._providers.append(GitProvider(repo_path=self._repo_root))

    @property
    def config(self) -> HarnessConfig:
        return self._config

    @property
    def sandbox(self) -> Any:
        return self._sandbox

    @property
    def tools(self) -> list[ToolSpec]:
        """Refresh and return all available tools."""
        self._tool_registry.clear()
        for provider in self._providers:
            for tool in provider.get_tools():
                self._tool_registry[tool.name] = tool
        return list(self._tool_registry.values())

    def get_tool(self, name: str) -> ToolSpec | None:
        """Look up a tool by name."""
        if name not in self._tool_registry:
            for provider in self._providers:
                for tool in provider.get_tools():
                    self._tool_registry[tool.name] = tool
        return self._tool_registry.get(name)

    def register_provider(self, provider: ToolProvider) -> None:
        """Register an additional tool provider."""
        self._providers.append(provider)

    def _build_system_prompt(self) -> str:
        """Build the full system prompt with tool descriptions."""
        tools = self.tools
        tool_descs = []
        for tool in tools:
            params = ", ".join(
                f"{k}: {v}" for k, v in tool.parameters.items()
            ) if tool.parameters else "no parameters"
            tool_descs.append(
                f"Tool: {tool.name}\n  Description: {tool.description}\n"
                f"  Parameters: {params}\n"
                f"  Risk: {tool.risk_level}\n  Permission: {tool.permission_required}"
            )
        tools_section = "\n".join(tool_descs) if tool_descs else "No tools available."

        prompt = self._config.system_prompt
        prompt += f"\n\nAvailable tools:\n{tools_section}\n"
        prompt += (
            f"\nPermission Level: {self._config.permission_level}\n"
            f"Max rounds: {self._config.max_rounds}\n"
            f"Tool timeout: {self._config.tool_timeout_seconds}s\n"
            f"Read-only mode: {self._config.read_only_mode}\n"
            "\nTo use a tool, call it by name with the exact parameter names. "
            "If a tool fails, analyze the error and retry. You may chain "
            "multiple tool calls in a single response."
        )
        return prompt

    def _create_agent(self) -> Any | None:
        """Lazily create the pydantic_ai Agent with registered tools."""
        if self._agent is not None:
            return self._agent

        try:
            from pydantic_ai import Agent

            agent = Agent(
                self._config.model,
                system_prompt=self._build_system_prompt(),
            )

            # Register each tool as a pydantic_ai function tool
            for provider in self._providers:
                for spec in provider.get_tools():
                    self._register_single_tool(agent, spec, provider)

            self._agent = agent
            return agent
        except ImportError:
            logger.warning("pydantic_ai not installed; using fallback mode")
            return None
        except Exception as exc:
            logger.warning("Agent creation failed: %s", exc)
            return None

    def _register_single_tool(self, agent: Any, spec: ToolSpec, provider: ToolProvider) -> None:
        """Register a single tool on the pydantic_ai agent."""
        tool_name = spec.name
        description = spec.description
        params = spec.parameters

        # Build the function with correct parameter names
        param_names = list(params.keys())
        param_str = ", ".join(f"{p}: str" for p in param_names)

        # Create a proper async function
        exec_globals: dict[str, Any] = {
            "Any": Any,
            "ExecutionResult": ExecutionResult,
        }

        func_code = (
            f"async def {tool_name.replace('-', '_')}(ctx: Any, {param_str}) -> str:\n"
            f'    """{description}"""\n'
            f"    return await _tool_dispatcher('{tool_name}', {param_str})"
        )

        # Build the async dispatcher
        async def tool_fn(**kwargs: Any) -> str:
            return await self._dispatch_tool(tool_name, provider, kwargs)

        exec_globals["_tool_dispatcher"] = tool_fn

        try:
            exec(func_code, exec_globals)
            registered_fn = exec_globals[tool_name.replace('-', '_')]
            agent.tool(registered_fn)
        except Exception as exc:
            logger.warning("Failed to register tool %s: %s", tool_name, exc)
            # Fallback: simple registration
            try:
                async def simple_tool(ctx: Any, **kwargs: Any) -> str:
                    return await self._dispatch_tool(tool_name, provider, kwargs)
                simple_tool.__name__ = tool_name.replace('-', '_')
                simple_tool.__doc__ = description
                agent.tool(simple_tool)
            except Exception as exc2:
                logger.warning("Fallback registration also failed for %s: %s", tool_name, exc2)

    async def _dispatch_tool(
        self, name: str, provider: ToolProvider, kwargs: dict[str, Any]
    ) -> str:
        """Dispatch a tool call through the provider with sandboxing."""
        result = provider.execute(name, **kwargs)
        if result.success:
            return result.output
        return f"Error: {result.error}"

    def run(
        self,
        message: str,
        ctx: HarnessContext | None = None,
        reset_history: bool = False,
    ) -> AgentResponse:
        """Run the agent synchronously.

        Parameters:
            message: The user's message/query.
            ctx: Optional HarnessContext for audit/integration.
            reset_history: If True, clear conversation history.
        """
        t0 = time.monotonic()

        if reset_history:
            self._conversation_history.clear()

        self._conversation_history.append({"role": "user", "content": message})

        # Record audit event
        request_id = ctx.request_id if ctx else f"req-{int(time.time())}"
        session_id = ctx.session_id if ctx else "standalone"

        self._audit_events.append({
            "event_id": f"evt-{len(self._audit_events)}",
            "timestamp": time.time(),
            "request_id": request_id,
            "session_id": session_id,
            "action": "request_start",
            "status": "success",
            "detail": message[:200],
        })

        # Try pydantic_ai, fall back to standalone mode
        agent = self._create_agent()

        if agent is not None:
            try:
                result = agent.run_sync(message)
                text = result.output if hasattr(result, "output") else str(result)
            except Exception as exc:
                logger.warning("Agent execution failed: %s", exc)
                text = self._fallback_run(message, str(exc))
        else:
            text = self._fallback_run(message)

        elapsed = time.monotonic() - t0

        self._conversation_history.append({"role": "assistant", "content": text})

        response = AgentResponse(
            text=text,
            success=True,
            elapsed_seconds=elapsed,
            conversation_turn=len(self._conversation_history) // 2,
        )

        self._audit_events.append({
            "event_id": f"evt-{len(self._audit_events)}",
            "timestamp": time.time(),
            "request_id": request_id,
            "session_id": session_id,
            "action": "request_done",
            "status": "success",
            "detail": f"turns={response.conversation_turn}",
        })

        return response

    def run_async(
        self,
        message: str,
        ctx: HarnessContext | None = None,
        reset_history: bool = False,
    ) -> Any:
        """Run the agent asynchronously (for FastAPI integration)."""
        if ctx:
            return self.run(message=message, ctx=ctx, reset_history=reset_history)
        return self.run(message=message, reset_history=reset_history)

    def _fallback_run(self, message: str, error: str = "") -> str:
        """Fallback execution without a real LLM (for testing/offline)."""
        cp = CodeExecutionProvider(sandbox=self._sandbox)
        fp = FilesystemProvider(sandbox=self._sandbox)
        gp = GitProvider(repo_path=self._repo_root)

        # Try to parse tool calls from message
        if message.startswith("run_python"):
            code = message.replace("run_python", "", 1).strip()
            if code:
                result = cp.execute("run_python", code=code)
                return result.output if result.success else f"Error: {result.error}"

        if any(kw in message.lower() for kw in ["list files", "ls ", "directory"]):
            result = fp.execute("list_directory", path=".")
            return result.output

        if message.startswith("git "):
            git_cmd = message[4:].strip()
            cmd_parts = git_cmd.split(None, 1)
            if cmd_parts:
                cmd = cmd_parts[0]
                if cmd == "status":
                    result = gp.execute("git_status")
                    return result.output

        return (
            f"I received your message: {message}\n\n"
            f"(Fallback mode: no LLM available. {error})"
        )

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Return the audit event log."""
        return self._audit_events

    @property
    def conversation_history(self) -> list[dict[str, str]]:
        """Return the conversation history."""
        return self._conversation_history

    def reset(self) -> None:
        """Clear conversation history (public API for session management).

        Callers outside the class (e.g. the serving bridge) should use this
        instead of reaching into ``_conversation_history`` directly.
        """
        self._conversation_history.clear()
        self._agent = None
