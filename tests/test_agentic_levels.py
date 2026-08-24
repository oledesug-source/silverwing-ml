"""Tests for the six-level agentic stack (offline, deterministic)."""

from __future__ import annotations

import pytest

from foundation.agentic.backend import DeterministicBackend, ScriptedBackend
from foundation.agentic.engine import AgenticEngine
from foundation.agentic.l1_l2 import RouteTarget, run_responder
from foundation.agentic.l3_to_l6 import (
    AgentRole,
    AutonomousGoalRunner,
    LoopEngineer,
    MultiAgentOrchestrator,
    SwPlatformToolRuntime,
    ToolCallingAgent,
)
from foundation.agentic.levels import AgentLevel


# ---------------------------------------------------------------- L1

def test_l1_responder_returns_generation() -> None:
    backend = ScriptedBackend(["hello there"])
    trace = run_responder(backend, "hi")
    assert trace.level is AgentLevel.BASIC_RESPONDER
    assert trace.success
    assert trace.final_text == "hello there"
    assert trace.steps[0].kind == "generate"


# ---------------------------------------------------------------- L2

def test_l2_router_dispatches_to_best_target() -> None:
    calls: list[str] = []

    def math_handler(message: str) -> str:
        calls.append("math")
        return f"math:{message}"

    router = __import__("foundation.agentic.l1_l2", fromlist=["IntentRouter"]).IntentRouter()
    router.register(RouteTarget(name="math", keywords=("calculate", "+", "sum"),
                                handler=math_handler))
    router.register(RouteTarget(
        name="chat", keywords=("hello",),
        handler=lambda m: f"chat:{m}",
    ))
    trace = router.handle("please calculate 2 + 3")
    assert calls == ["math"]
    assert trace.final_text.startswith("math:")
    assert trace.steps[0].data["confidence"] > 0


def test_l2_router_falls_back_when_no_route_matches() -> None:
    from foundation.agentic.l1_l2 import IntentRouter

    router = IntentRouter(default=lambda m: f"fallback:{m}")
    trace = router.handle("zzz unrelated")
    assert trace.final_text == "fallback:zzz unrelated"


# ---------------------------------------------------------------- L3

class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def catalog(self) -> list[dict]:
        return [{"name": "get_time", "description": "current time",
                 "parameters": {"tz": "str"}}]

    def call(self, name: str, kwargs: dict) -> list[dict]:
        self.calls.append((name, kwargs))
        return [{"tool": name, "success": True,
                 "output": "12:00", "error": ""}]


def test_l3_tool_calling_executes_tool_then_final() -> None:
    runtime = _FakeRuntime()
    backend = ScriptedBackend([
        'TOOL: get_time {"tz": "utc"}',
        "FINAL: it is 12:00",
    ])
    agent = ToolCallingAgent(backend, runtime)
    trace = agent.run("what time is it?")
    assert runtime.calls == [("get_time", {"tz": "utc"})]
    assert [s.kind for s in trace.steps] == ["tool_call", "final"]
    assert trace.final_text == "it is 12:00"
    assert trace.success


def test_l3_tool_calling_reports_max_rounds() -> None:
    class LoopRuntime(_FakeRuntime):
        pass

    backend = ScriptedBackend(['TOOL: get_time {}'])
    agent = ToolCallingAgent(backend, _FakeRuntime(), max_rounds=2)
    trace = agent.run("loop forever")
    # scripted backend repeats last response; every round returns a tool call
    assert not trace.success
    assert trace.final_text == "max tool rounds reached"


def test_sw_platform_runtime_unknown_tool_is_reported() -> None:
    class _P:
        def get_tools(self):
            class _S:
                name = "known"
                description = ""
                parameters = {}
            return [_S()]

        def execute(self, name, **kwargs):
            raise AssertionError("should not be called")

    runtime = SwPlatformToolRuntime([_P()])
    result = runtime.call("unknown", {})
    assert result[0]["success"] is False
    assert "unknown tool" in result[0]["error"]


# ---------------------------------------------------------------- L4

def test_l4_multi_agent_routes_by_keywords() -> None:
    backend = ScriptedBackend(["code answer"])
    roles = [
        AgentRole(name="researcher", system_prompt="r", keywords=("research",)),
        AgentRole(name="coder", system_prompt="c", keywords=("python",)),
    ]
    orchestrator = MultiAgentOrchestrator(backend, roles)
    trace = orchestrator.run("write python code")
    assert trace.data["agents_engaged"] == ["coder"]
    assert "[coder] code answer" in trace.final_text


def test_l4_multi_agent_requires_roles() -> None:
    with pytest.raises(ValueError):
        MultiAgentOrchestrator(DeterministicBackend(), [])


# ---------------------------------------------------------------- L5

def test_l5_autonomous_executes_full_plan() -> None:
    executed: list[str] = []
    backend = ScriptedBackend([
        "1. step one\n2. step two\n3. step three",
    ])
    runner = AutonomousGoalRunner(backend, executed.append)
    trace = runner.run("build a thing")
    assert len(executed) == 3
    assert trace.success
    assert "3/3 steps completed" in trace.final_text


def test_l5_autonomous_approval_gate_aborts() -> None:
    backend = ScriptedBackend(["1. safe\n2. dangerous"])
    runner = AutonomousGoalRunner(
        backend, lambda s: s,
        approve=lambda step: "dangerous" not in step.description,
    )
    trace = runner.run("goal")
    assert not trace.success
    assert "aborted at approval gate" in trace.final_text


# ---------------------------------------------------------------- L6

def test_l6_loop_converges_on_first_reflection() -> None:
    backend = DeterministicBackend()
    loop_engineer = LoopEngineer(backend, lambda directive: f"did: {directive[:20]}")
    trace = loop_engineer.run("polish the docs")
    assert trace.success
    assert trace.data["converged"]
    assert trace.data["cycles"] == 1


def test_l6_loop_retries_until_budget() -> None:
    responses = ["CRITIQUE: not yet.\nGOAL_MET: no"] * 5
    backend = ScriptedBackend(responses)
    runs: list[str] = []

    def inner(directive: str) -> str:
        runs.append(directive)
        return "attempt"

    loop_engineer = LoopEngineer(backend, inner, max_cycles=3)
    trace = loop_engineer.run("hard goal")
    assert len(runs) == 3
    assert not trace.success
    assert trace.data["memory"]  # reflections recorded


# ---------------------------------------------------------------- engine

def test_engine_runs_every_level() -> None:
    engine = AgenticEngine(DeterministicBackend())
    for level in AgentLevel:
        trace = engine.run(level, "explain python")
        assert trace.level is level
        assert isinstance(trace.final_text, str)


def test_engine_custom_route_beats_default() -> None:
    engine = AgenticEngine(DeterministicBackend())
    engine.register_route(RouteTarget(
        name="special", keywords=("special",),
        handler=lambda m: "handled-specially"))
    trace = engine.run(AgentLevel.ROUTER, "a special request")
    assert trace.final_text == "handled-specially"
