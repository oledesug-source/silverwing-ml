"""AgenticEngine — one facade over the six capability levels."""

from __future__ import annotations

import os
import time
from typing import Any

from .backend import DeterministicBackend, HttpOpenAICompat, LlmBackend
from .l1_l2 import IntentRouter, RouteTarget, run_responder
from .l3_to_l6 import (
    AgentRole,
    AutonomousGoalRunner,
    LoopEngineer,
    MultiAgentOrchestrator,
    SwPlatformToolRuntime,
    ToolCallingAgent,
)
from .levels import AgentLevel, AgentTrace, coerce_level

DEFAULT_ROLES = [
    AgentRole(
        name="researcher",
        system_prompt="You research facts and summarise evidence concisely.",
        keywords=("research", "find", "look up", "explain", "what is", "who"),
        priority=1,
    ),
    AgentRole(
        name="coder",
        system_prompt="You write correct, minimal code and explain trade-offs.",
        keywords=("code", "python", "script", "function", "bug", "implement"),
        priority=2,
    ),
    AgentRole(
        name="critic",
        system_prompt="You critique plans for risks, gaps and hidden assumptions.",
        keywords=("review", "risk", "critique", "assess"),
        priority=0,
    ),
]


def _make_backend() -> LlmBackend:
    model_env = os.environ.get("SILVERWING_AGENT_MODEL")
    if model_env:
        return HttpOpenAICompat(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            model=model_env,
            api_key=os.environ.get("OPENAI_API_KEY", "local"),
        )
    return DeterministicBackend()


def _platform_providers(allowed_paths: list[str] | None) -> list[Any]:
    """Best-effort construction of sw_platform tool providers."""
    try:
        from sw_platform.sandbox.executor import ResourceLimits, SandboxExecutor
        from sw_platform.tools.code_execution import CodeExecutionProvider
        from sw_platform.tools.filesystem import FilesystemProvider

        sandbox = SandboxExecutor(limits=ResourceLimits(
            max_execution_time=30.0,
            network_allowed=False,
            allowed_paths=allowed_paths or [],
        ))
        providers: list[Any] = [
            CodeExecutionProvider(sandbox=sandbox),
            FilesystemProvider(allowed_paths=allowed_paths or [], sandbox=sandbox),
        ]
        return providers
    except Exception:
        return []


class _ProviderToolRuntime:
    """ToolRuntime over already-constructed providers (no lazy imports)."""

    def __init__(self, providers: list[Any]) -> None:
        from .l3_to_l6 import SwPlatformToolRuntime

        self._inner = SwPlatformToolRuntime(providers)

    def catalog(self) -> list[dict[str, Any]]:
        return self._inner.catalog()

    def call(self, name: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        return self._inner.call(name, kwargs)


class AgenticEngine:
    """Dispatch a message to any of the six levels and return a full trace.

    Usage::

        engine = AgenticEngine()
        trace = engine.run(AgentLevel.TOOL_CALLING, "list files in .")
        print(trace.final_text, [s.kind for s in trace.steps])
    """

    def __init__(
        self,
        backend: LlmBackend | None = None,
        *,
        allowed_paths: list[str] | None = None,
        roles: list[AgentRole] | None = None,
        max_rounds: int = 5,
        max_cycles: int = 4,
    ) -> None:
        self.backend = backend or _make_backend()
        self.allowed_paths = allowed_paths
        self.roles = roles or list(DEFAULT_ROLES)
        self.max_rounds = max_rounds
        self.max_cycles = max_cycles
        self._providers: list[Any] | None = None
        self._router: IntentRouter | None = None

    def register_route(self, target: RouteTarget) -> None:
        self.router.register(target)

    @property
    def router(self) -> IntentRouter:
        if self._router is None:
            self._router = IntentRouter(default=lambda m: run_responder(
                self.backend, m).final_text)
        return self._router

    def tool_runtime(self) -> Any:
        if self._providers is None:
            try:
                self._providers = _platform_providers(self.allowed_paths)
            except Exception:
                self._providers = []
        return _ProviderToolRuntime(self._providers)

    def tool_catalog(self) -> list[dict[str, Any]]:
        return self.tool_runtime().catalog()

    def run(
        self,
        level: int | str | AgentLevel,
        message: str,
        *,
        session_id: str = "",
        max_tokens: int = 512,
    ) -> AgentTrace:
        resolved = coerce_level(level)
        t0 = time.monotonic()
        runner = self._runner_for(resolved, max_tokens)
        trace = runner(message)
        trace.session_id = session_id
        trace.elapsed_seconds = time.monotonic() - t0
        return trace

    def _runner_for(self, level: AgentLevel, max_tokens: int):
        if level is AgentLevel.BASIC_RESPONDER:
            return lambda m: run_responder(self.backend, m, max_tokens=max_tokens)
        if level is AgentLevel.ROUTER:
            return self.router.handle
        if level is AgentLevel.TOOL_CALLING:
            agent = ToolCallingAgent(self.backend, self.tool_runtime(),
                                     max_rounds=self.max_rounds)
            return agent.run
        if level is AgentLevel.MULTI_AGENT:
            orchestrator = MultiAgentOrchestrator(self.backend, self.roles)
            return orchestrator.run
        if level is AgentLevel.AUTONOMOUS:
            inner_tool_agent = ToolCallingAgent(self.backend, self.tool_runtime(),
                                                max_rounds=self.max_rounds)

            def step_runner(step_text: str) -> str:
                return inner_tool_agent.run(step_text).final_text

            runner_obj = AutonomousGoalRunner(self.backend, step_runner)
            return runner_obj.run
        if level is AgentLevel.LOOP_ENGINEERING:
            autonomous = AutonomousGoalRunner(
                self.backend,
                lambda s: ToolCallingAgent(self.backend, self.tool_runtime(),
                                           max_rounds=self.max_rounds).run(s).final_text,
            )
            loop_engineer = LoopEngineer(
                self.backend,
                lambda goal: autonomous.run(goal).final_text,
                max_cycles=self.max_cycles,
            )
            return loop_engineer.run
        raise ValueError(f"unhandled level {level}")


def create_default_engine(**kwargs: Any) -> AgenticEngine:
    """Engine wired to env-configured backend + platform tools."""
    return AgenticEngine(**kwargs)
