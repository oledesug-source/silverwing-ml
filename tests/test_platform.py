"""Comprehensive tests for the platform/ package.

All tests run without torch — mock generator is used for orchestration tests.
"""

from __future__ import annotations

import ast
import operator
from typing import Any
from unittest.mock import MagicMock

from intelligence.tools.protocol import ToolCall, ToolResult
from sw_platform.audit.events import AuditEvent, AuditLog
from sw_platform.capabilities.discovery import CapabilityDiscovery
from sw_platform.capabilities.registry import CapabilityRegistry
from sw_platform.capabilities.schema import CapabilitySchema
from sw_platform.context.builder import ContextBuilder
from sw_platform.context.models import SessionState
from sw_platform.orchestration.execution_loop import ExecutionLoop
from sw_platform.orchestration.orchestrator import (
    ChatRequest,
    Orchestrator,
)
from sw_platform.permissions.policy import (
    PermissionEvaluator,
    PermissionLevel,
    PermissionPolicy,
)
from sw_platform.sandbox.executor import ResourceLimits, SandboxExecutor

# ======================================================================
# Helpers
# ======================================================================

def _make_cap(
    name: str = "calculator",
    fn: Any = None,
    tags: list[str] | None = None,
    enabled: bool = True,
    risk_level: str = "low",
    permissions_required: list[str] | None = None,
    description: str = "A test capability",
    input_schema: dict | None = None,
    capability_type: str = "tool",
) -> CapabilitySchema:
    return CapabilitySchema(
        name=name,
        fn=fn,
        tags=tags or [],
        enabled=enabled,
        risk_level=risk_level,
        permissions_required=permissions_required or ["L0"],
        description=description,
        input_schema=input_schema or {},
        capability_type=capability_type,
    )


def _mock_generator(output: str) -> MagicMock:
    gen = MagicMock()
    result = MagicMock()
    result.text = output
    gen.generate.return_value = result
    return gen


def _mock_generator_sequence(outputs: list[str]) -> MagicMock:
    gen = MagicMock()
    results = [MagicMock(text=o) for o in outputs]
    gen.generate.side_effect = results
    return gen


def _safe_calc(expression: str) -> str:
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


# ======================================================================
# TestCapabilitySchema
# ======================================================================

class TestCapabilitySchema:
    def test_creation(self):
        cap = _make_cap(name="test")
        assert cap.name == "test"
        assert cap.version == "1.0.0"
        assert cap.enabled is True
        assert cap.risk_level == "low"

    def test_defaults(self):
        cap = _make_cap()
        assert cap.id  # auto-generated
        assert cap.timeout_seconds == 30.0
        assert cap.execution_mode == "sync"
        assert cap.capability_type == "tool"
        assert cap.source == "builtin"
        assert cap.permissions_required == ["L0"]

    def test_matches_permission_low(self):
        cap = _make_cap(permissions_required=["L0"])
        assert cap.matches_permission("L0")
        assert cap.matches_permission("L5")

    def test_matches_permission_high(self):
        cap = _make_cap(permissions_required=["L3"])
        assert not cap.matches_permission("L0")
        assert not cap.matches_permission("L2")
        assert cap.matches_permission("L3")
        assert cap.matches_permission("L5")


# ======================================================================
# TestCapabilityRegistry
# ======================================================================

class TestCapabilityRegistry:
    def test_register_and_get(self):
        reg = CapabilityRegistry()
        cap = _make_cap(name="calc")
        reg.register(cap)
        assert reg.get("calc") is cap

    def test_get_unknown(self):
        reg = CapabilityRegistry()
        assert reg.get("nope") is None

    def test_unregister(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="x"))
        assert reg.unregister("x") is True
        assert reg.get("x") is None
        assert reg.unregister("x") is False

    def test_list_all(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="a"))
        reg.register(_make_cap(name="b"))
        assert len(reg.list()) == 2

    def test_list_enabled_only(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="a", enabled=True))
        reg.register(_make_cap(name="b", enabled=False))
        assert len(reg.list(enabled_only=True)) == 1
        assert reg.list(enabled_only=True)[0].name == "a"

    def test_search_by_name(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calculator", description="Math tool"))
        reg.register(_make_cap(name="reader", description="File reader"))
        results = reg.search(query="calc")
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_search_by_tags(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="a", tags=["math"]))
        reg.register(_make_cap(name="b", tags=["file"]))
        results = reg.search(tags=["math"])
        assert len(results) == 1
        assert results[0].name == "a"

    def test_search_by_type(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="a", capability_type="tool"))
        reg.register(_make_cap(name="b", capability_type="reasoning"))
        results = reg.search(capability_type="reasoning")
        assert len(results) == 1
        assert results[0].name == "b"

    def test_enable_disable(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="x"))
        assert reg.disable("x") is True
        assert len(reg.list(enabled_only=True)) == 0
        assert reg.enable("x") is True
        assert len(reg.list(enabled_only=True)) == 1

    def test_system_prompt(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", description="Do math"))
        prompt = reg.system_prompt()
        assert "calculator" in prompt.lower() or "calc" in prompt.lower()

    def test_parse_calls(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc"))
        text = "Let me use <tool:calc>2+2</tool> to solve this."
        calls = reg.parse_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "calc"

    def test_execute_call(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))
        call = ToolCall(tool_name="calc", arguments="expression=2+3")
        result = reg.execute_call(call)
        assert result.success
        assert result.output == "5"

    def test_execute_unknown(self):
        reg = CapabilityRegistry()
        call = ToolCall(tool_name="nope", arguments="")
        result = reg.execute_call(call)
        assert not result.success

    def test_format_results(self):
        reg = CapabilityRegistry()
        results = [
            ToolResult(tool_name="a", output="ok", success=True),
            ToolResult(tool_name="b", output="", success=False, error="fail"),
        ]
        text = reg.format_results(results)
        assert "a" in text
        assert "ok" in text
        assert "b" in text
        assert "fail" in text

    def test_to_tool_registry(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="x"))
        tr = reg.to_tool_registry()
        assert tr.get("x") is not None


# ======================================================================
# TestCapabilityDiscovery
# ======================================================================

class TestCapabilityDiscovery:
    def test_find_for_task(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calculator", tags=["math", "calculate"], description="Evaluate math expressions"))
        reg.register(_make_cap(name="read_file", tags=["file", "read"], description="Read files from disk"))
        discovery = CapabilityDiscovery(reg)
        results = discovery.find_for_task("calculate math problem")
        assert len(results) >= 1
        assert results[0].name == "calculator"

    def test_no_match(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calculator", tags=["math"]))
        discovery = CapabilityDiscovery(reg)
        results = discovery.find_for_task("play music")
        assert len(results) == 0

    def test_disabled_excluded(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calculator", tags=["math"], enabled=False))
        discovery = CapabilityDiscovery(reg)
        results = discovery.find_for_task("calculate math")
        assert len(results) == 0


# ======================================================================
# TestSessionState
# ======================================================================

class TestSessionState:
    def test_creation(self):
        s = SessionState()
        assert s.session_id
        assert s.user_id == ""
        assert len(s.working_memory) == 0

    def test_with_user_id(self):
        s = SessionState(user_id="user-42")
        assert s.user_id == "user-42"


# ======================================================================
# TestRequestContext
# ======================================================================

class TestRequestContext:
    def test_creation(self):
        ctx = ContextBuilder.from_request("hello")
        assert ctx.user_message == "hello"
        assert ctx.request_id
        assert ctx.max_rounds == 5

    def test_add_user_message(self):
        ctx = ContextBuilder.from_request("hello")
        ctx.add_user_message()
        assert len(ctx.working_memory) == 1

    def test_add_tool_result_success(self):
        ctx = ContextBuilder.from_request("test")
        result = ToolResult(tool_name="calc", output="42", success=True)
        ctx.add_tool_result(result)
        assert len(ctx.tool_results) == 1
        assert "calc" in ctx.capabilities_used
        assert len(ctx.working_memory) == 1

    def test_add_tool_result_error(self):
        ctx = ContextBuilder.from_request("test")
        result = ToolResult(tool_name="calc", output="", success=False, error="bad")
        ctx.add_tool_result(result)
        assert len(ctx.tool_results) == 1

    def test_add_assistant_message(self):
        ctx = ContextBuilder.from_request("test")
        ctx.add_assistant_message("response")
        assert len(ctx.working_memory) == 1


# ======================================================================
# TestContextBuilder
# ======================================================================

class TestContextBuilder:
    def test_from_request(self):
        ctx = ContextBuilder.from_request("hi", max_rounds=3, user_id="u1")
        assert ctx.user_message == "hi"
        assert ctx.max_rounds == 3
        assert ctx.session.user_id == "u1"

    def test_build_system_prompt(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", description="Math"))
        prompt = ContextBuilder.build_system_prompt(reg, "L0")
        assert "Permission level" in prompt
        assert "calc" in prompt.lower() or "math" in prompt.lower()

    def test_build_system_prompt_empty(self):
        reg = CapabilityRegistry()
        prompt = ContextBuilder.build_system_prompt(reg, "L0")
        assert "No capabilities" in prompt


# ======================================================================
# TestPermissionLevel
# ======================================================================

class TestPermissionLevel:
    def test_ordering(self):
        assert PermissionLevel.L0 < PermissionLevel.L1
        assert PermissionLevel.L2 < PermissionLevel.L5
        assert PermissionLevel.L5 >= PermissionLevel.L0
        assert PermissionLevel.L3 == PermissionLevel.L3

    def test_numeric(self):
        assert PermissionLevel.L0.numeric == 0
        assert PermissionLevel.L5.numeric == 5

    def test_value(self):
        assert PermissionLevel.L0.value == "read"
        assert PermissionLevel.L3.value == "network"


# ======================================================================
# TestPermissionPolicy
# ======================================================================

class TestPermissionPolicy:
    def test_default(self):
        p = PermissionPolicy()
        assert p.level == PermissionLevel.L0
        assert p.allowed_tools is None
        assert len(p.denied_tools) == 0

    def test_with_level(self):
        p = PermissionPolicy(level=PermissionLevel.L2)
        assert p.level == PermissionLevel.L2


# ======================================================================
# TestPermissionEvaluator
# ======================================================================

class TestPermissionEvaluator:
    def test_allowed(self):
        policy = PermissionPolicy(level=PermissionLevel.L0)
        ev = PermissionEvaluator(policy)
        cap = _make_cap(permissions_required=["L0"])
        ok, reason = ev.is_allowed(cap)
        assert ok

    def test_insufficient(self):
        policy = PermissionPolicy(level=PermissionLevel.L0)
        ev = PermissionEvaluator(policy)
        cap = _make_cap(permissions_required=["L2"])
        ok, reason = ev.is_allowed(cap)
        assert not ok
        assert "Insufficient" in reason

    def test_disabled_denied(self):
        policy = PermissionPolicy(level=PermissionLevel.L5)
        ev = PermissionEvaluator(policy)
        cap = _make_cap(enabled=False)
        ok, reason = ev.is_allowed(cap)
        assert not ok
        assert "disabled" in reason

    def test_denied_tool(self):
        policy = PermissionPolicy(level=PermissionLevel.L5, denied_tools={"shell"})
        ev = PermissionEvaluator(policy)
        cap = _make_cap(name="shell")
        ok, reason = ev.is_allowed(cap)
        assert not ok
        assert "denied" in reason

    def test_allowed_tools_whitelist(self):
        policy = PermissionPolicy(level=PermissionLevel.L5, allowed_tools={"calc"})
        ev = PermissionEvaluator(policy)
        cap = _make_cap(name="other")
        ok, reason = ev.is_allowed(cap)
        assert not ok
        assert "not in the allowed list" in reason

    def test_needs_sandbox_high_risk(self):
        policy = PermissionPolicy()
        ev = PermissionEvaluator(policy)
        cap = _make_cap(risk_level="high")
        assert ev.needs_sandbox(cap)

    def test_needs_sandbox_explicit(self):
        policy = PermissionPolicy(require_sandbox={"shell"})
        ev = PermissionEvaluator(policy)
        cap = _make_cap(name="shell")
        assert ev.needs_sandbox(cap)

    def test_get_max_permission_level(self):
        policy = PermissionPolicy(level=PermissionLevel.L3)
        ev = PermissionEvaluator(policy)
        assert ev.get_max_permission_level() == PermissionLevel.L3


# ======================================================================
# TestSandboxExecutor
# ======================================================================

class TestSandboxExecutor:
    def test_success(self):
        sb = SandboxExecutor(ResourceLimits(max_execution_time=5.0))
        result = sb.execute(lambda expression: _safe_calc(expression), cap_id="calc", expression="2+3")
        assert result.success
        assert result.output == "5"

    def test_error(self):
        sb = SandboxExecutor()
        result = sb.execute(lambda: 1 / 0, cap_id="div")
        assert not result.success
        assert "error" in result.error.lower() or "division" in result.error.lower()

    def test_timeout(self):
        import time as _time

        def slow():
            _time.sleep(10)

        sb = SandboxExecutor(ResourceLimits(max_execution_time=0.1))
        result = sb.execute(slow, cap_id="slow")
        assert not result.success
        assert "timeout" in result.error.lower()

    def test_path_check_blocked(self):
        sb = SandboxExecutor(ResourceLimits(blocked_paths=["/etc"]))
        ok, reason = sb.check_path("/etc/passwd")
        assert not ok
        assert "blocked" in reason.lower()

    def test_path_check_allowed(self):
        sb = SandboxExecutor(ResourceLimits(allowed_paths=["/home/user"]))
        ok, _ = sb.check_path("/home/user/doc.txt")
        assert ok

    def test_path_check_not_in_allowed(self):
        sb = SandboxExecutor(ResourceLimits(allowed_paths=["/home/user"]))
        ok, reason = sb.check_path("/tmp/file.txt")
        assert not ok
        assert "not in allowed" in reason.lower()

    def test_file_size_ok(self):
        sb = SandboxExecutor(ResourceLimits(max_file_size_bytes=1000))
        ok, _ = sb.check_file_size(500)
        assert ok

    def test_file_size_too_large(self):
        sb = SandboxExecutor(ResourceLimits(max_file_size_bytes=100))
        ok, reason = sb.check_file_size(200)
        assert not ok
        assert "too large" in reason.lower()


# ======================================================================
# TestAuditEvent
# ======================================================================

class TestAuditEvent:
    def test_creation(self):
        e = AuditEvent(action="test")
        assert e.event_id
        assert e.timestamp > 0
        assert e.status == "pending"

    def test_to_dict(self):
        e = AuditEvent(action="tool_call", capability_id="calc")
        d = e.to_dict()
        assert d["action"] == "tool_call"
        assert d["capability_id"] == "calc"


# ======================================================================
# TestAuditLog
# ======================================================================

class TestAuditLog:
    def test_record_and_query(self):
        log = AuditLog()
        log.record(AuditEvent(request_id="r1", action="start"))
        log.record(AuditEvent(request_id="r2", action="end"))
        assert len(log) == 2

        results = log.query(request_id="r1")
        assert len(results) == 1

    def test_recent(self):
        log = AuditLog()
        for i in range(10):
            log.record(AuditEvent(action=f"event_{i}"))
        recent = log.recent(3)
        assert len(recent) == 3

    def test_overflow(self):
        log = AuditLog(max_entries=5)
        for i in range(10):
            log.record(AuditEvent(action=f"event_{i}"))
        assert len(log) == 5
        assert log.recent(1)[0].action == "event_9"

    def test_clear(self):
        log = AuditLog()
        log.record(AuditEvent(action="x"))
        log.clear()
        assert len(log) == 0

    def test_query_by_status(self):
        log = AuditLog()
        log.record(AuditEvent(action="a", status="success"))
        log.record(AuditEvent(action="b", status="error"))
        assert len(log.query(status="success")) == 1
        assert len(log.query(status="error")) == 1


# ======================================================================
# TestExecutionLoop
# ======================================================================

class TestExecutionLoop:
    def test_no_tool_calls(self):
        loop = ExecutionLoop(max_steps=5)
        ctx = ContextBuilder.from_request("hello")

        def step(ctx, override):
            return "No tools here."

        text, calls, results, rounds = loop.run(
            step_fn=step, context=ctx,
            registry=CapabilityRegistry(),
            evaluator=PermissionEvaluator(PermissionPolicy()),
            sandbox=SandboxExecutor(),
        )
        assert text == "No tools here."
        assert calls == []
        assert rounds == 0

    def test_single_tool_call(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))

        loop = ExecutionLoop(max_steps=5)
        ctx = ContextBuilder.from_request("<tool:calc>expression=2+2</tool>")

        call_count = [0]
        def step(ctx, override):
            call_count[0] += 1
            if call_count[0] == 1:
                return "<tool:calc>expression=2+2</tool>"
            return "The answer is 4."

        text, calls, results, rounds = loop.run(
            step_fn=step, context=ctx,
            registry=reg,
            evaluator=PermissionEvaluator(PermissionPolicy()),
            sandbox=SandboxExecutor(),
        )
        assert len(calls) == 1
        assert results[0].output == "4"
        assert "answer" in text.lower() or "4" in text

    def test_max_steps_limit(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))

        loop = ExecutionLoop(max_steps=2)
        ctx = ContextBuilder.from_request("calc")

        def step(ctx, override):
            return "<tool:calc>expression=1+1</tool>"

        text, calls, results, rounds = loop.run(
            step_fn=step, context=ctx,
            registry=reg,
            evaluator=PermissionEvaluator(PermissionPolicy()),
            sandbox=SandboxExecutor(),
        )
        assert rounds == 2
        assert len(calls) == 2

    def test_permission_denied(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="shell", permissions_required=["L3"]))

        loop = ExecutionLoop(max_steps=3)
        ctx = ContextBuilder.from_request("shell")

        call_count = [0]
        def step(ctx, override):
            call_count[0] += 1
            return "<tool:shell>command=ls</tool>"

        text, calls, results, rounds = loop.run(
            step_fn=step, context=ctx,
            registry=reg,
            evaluator=PermissionEvaluator(PermissionPolicy(level=PermissionLevel.L0)),
            sandbox=SandboxExecutor(),
        )
        assert all(not r.success for r in results)
        assert all("Insufficient" in r.error for r in results)
        assert rounds == 3


# ======================================================================
# TestOrchestrator
# ======================================================================

class TestOrchestrator:
    def test_simple_response(self):
        orch = Orchestrator(registry=CapabilityRegistry())
        response = orch.handle_request(ChatRequest(message="hello"))
        assert response.success
        assert "hello" in response.text.lower()

    def test_with_tool_call(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))
        gen = _mock_generator_sequence([
            "<tool:calc>expression=3*4</tool>",
            "The answer is 12.",
        ])
        orch = Orchestrator(registry=reg, generator=gen)
        response = orch.handle_request(ChatRequest(message="what is 3*4?"))
        assert response.success
        assert len(response.tool_calls) == 1
        assert response.tool_results[0].output == "12"
        assert response.rounds == 1

    def test_multi_round(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))
        gen = _mock_generator_sequence([
            "<tool:calc>expression=2+2</tool>",
            "<tool:calc>expression=3+3</tool>",
            "Results: 4 and 6.",
        ])
        orch = Orchestrator(registry=reg, generator=gen)
        response = orch.handle_request(ChatRequest(message="math"))
        assert response.rounds == 2
        assert len(response.tool_calls) == 2

    def test_permission_denied(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="shell", permissions_required=["L3"]))
        gen = _mock_generator("<tool:shell>command=ls</tool>")
        orch = Orchestrator(
            registry=reg, generator=gen,
            permissions=PermissionPolicy(level=PermissionLevel.L0),
        )
        response = orch.handle_request(ChatRequest(message="run shell"))
        assert not response.tool_results[0].success

    def test_no_generator_fallback(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))
        orch = Orchestrator(registry=reg)
        response = orch.handle_request(
            ChatRequest(message="<tool:calc>expression=5+5</tool>"),
        )
        assert response.success
        assert "10" in response.text

    def test_no_generator_plain_text(self):
        orch = Orchestrator(registry=CapabilityRegistry())
        response = orch.handle_request(ChatRequest(message="hello"))
        assert response.success
        assert "hello" in response.text

    def test_list_capabilities(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc"))
        orch = Orchestrator(registry=reg)
        caps = orch.list_capabilities()
        assert len(caps) == 1
        assert caps[0]["name"] == "calc"

    def test_response_to_dict(self):
        orch = Orchestrator(registry=CapabilityRegistry())
        response = orch.handle_request(ChatRequest(message="hi"))
        d = response.to_dict()
        assert "text" in d
        assert "success" in d
        assert "tool_calls" in d
        assert "audit_events" in d

    def test_elapsed_seconds(self):
        orch = Orchestrator(registry=CapabilityRegistry())
        response = orch.handle_request(ChatRequest(message="hi"))
        assert response.elapsed_seconds >= 0

    def test_request_id(self):
        orch = Orchestrator(registry=CapabilityRegistry())
        response = orch.handle_request(ChatRequest(message="hi"))
        assert response.request_id

    def test_audit_trail(self):
        orch = Orchestrator(registry=CapabilityRegistry())
        response = orch.handle_request(ChatRequest(message="hi"))
        assert len(response.audit_events) >= 2  # request_start + request_done

    def test_round_limit(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(name="calc", fn=lambda expression: _safe_calc(expression)))
        gen = _mock_generator("<tool:calc>expression=1+1</tool>")
        orch = Orchestrator(registry=reg, generator=gen, max_steps=1)
        response = orch.handle_request(ChatRequest(message="math"))
        assert response.rounds == 1


# ======================================================================
# TestPlatformPublicAPI
# ======================================================================

class TestPlatformPublicAPI:
    def test_imports(self):
        from sw_platform import (
            CapabilityRegistry,
            Orchestrator,
            PermissionLevel,
        )
        assert CapabilityRegistry is not None
        assert Orchestrator is not None
        assert PermissionLevel.L0 is not None
