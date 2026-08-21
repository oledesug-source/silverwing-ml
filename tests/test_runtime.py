"""Tests for the runtime/ package (Intelligence Runtime v1).

Covers: capabilities, context, permissions, policies, sandbox,
orchestration, agents, tools, and workflows.
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.memory.context import WorkingMemory
from intelligence.tools.protocol import (
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from runtime.agents import Agent
from runtime.capabilities import Capability, CapabilityRegistry
from runtime.context import RequestContext
from runtime.orchestration import (
    ChatRequest,
    ChatResponse,
    Orchestrator,
)
from runtime.permissions import PermissionCheck, PermissionPolicy
from runtime.policies import PolicyEngine
from runtime.sandbox import Sandbox
from runtime.tools import calculator, read_file, register_builtin_tools
from runtime.workflows import StepResult, Workflow, WorkflowResult, WorkflowStep

# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------

class TestCapability:
    def test_creation_all_fields(self):
        cap = Capability(
            name="calc",
            description="math",
            parameters={"expression": "str"},
            fn=lambda expression: "42",
            source="builtin",
            tags=["math"],
            requires_permission=True,
            timeout_seconds=10.0,
        )
        assert cap.name == "calc"
        assert cap.description == "math"
        assert cap.parameters == {"expression": "str"}
        assert cap.fn is not None
        assert cap.source == "builtin"
        assert cap.tags == ["math"]
        assert cap.requires_permission is True
        assert cap.timeout_seconds == 10.0

    def test_defaults(self):
        cap = Capability(name="x", description="d")
        assert cap.parameters == {}
        assert cap.fn is None
        assert cap.source == "builtin"
        assert cap.tags == []
        assert cap.requires_permission is False
        assert cap.timeout_seconds == 30.0

    def test_to_tool(self):
        cap = Capability(
            name="calc",
            description="math",
            parameters={"expression": "str"},
            fn=lambda expression: "42",
        )
        tool = cap.to_tool()
        assert isinstance(tool, Tool)
        assert tool.name == "calc"
        assert tool.description == "math"
        assert tool.parameters == {"expression": "str"}
        assert tool.fn is not None

    def test_to_tool_no_fn(self):
        cap = Capability(name="x", description="d")
        tool = cap.to_tool()
        assert isinstance(tool, Tool)
        assert tool.fn is None
        assert tool.parameters == {}

    def test_to_tool_empty_parameters(self):
        cap = Capability(name="x", description="d", parameters={})
        tool = cap.to_tool()
        assert tool.parameters == {}


# ---------------------------------------------------------------------------
# CapabilityRegistry
# ---------------------------------------------------------------------------

class TestCapabilityRegistry:
    def test_register(self):
        reg = CapabilityRegistry()
        cap = Capability(name="calc", description="math", fn=lambda: "ok")
        reg.register(cap)
        assert reg.get("calc") is cap

    def test_register_multiple(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", description="d"))
        reg.register(Capability(name="b", description="d"))
        caps = reg.list_capabilities()
        assert len(caps) == 2

    def test_get_not_found(self):
        reg = CapabilityRegistry()
        assert reg.get("nonexistent") is None

    def test_discover_all(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", description="d", tags=["math"]))
        reg.register(Capability(name="b", description="d", tags=["file"]))
        all_caps = reg.discover()
        assert len(all_caps) == 2

    def test_discover_by_tag(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", description="d", tags=["math"]))
        reg.register(Capability(name="b", description="d", tags=["file"]))
        math_caps = reg.discover(tags=["math"])
        assert len(math_caps) == 1
        assert math_caps[0].name == "a"

    def test_discover_multiple_tags_any_match(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", description="d", tags=["math", "safe"]))
        reg.register(Capability(name="b", description="d", tags=["file"]))
        matches = reg.discover(tags=["math", "file"])
        assert len(matches) == 2

    def test_discover_no_matches(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", description="d", tags=["math"]))
        matches = reg.discover(tags=["nonexistent"])
        assert len(matches) == 0

    def test_list_capabilities(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="a", description="d"))
        reg.register(Capability(name="b", description="d"))
        caps = reg.list_capabilities()
        names = [c.name for c in caps]
        assert "a" in names
        assert "b" in names

    def test_system_prompt(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="calc", description="math", parameters={"expression": "str"}))
        prompt = reg.system_prompt()
        assert "calc" in prompt
        assert "math" in prompt

    def test_system_prompt_empty(self):
        reg = CapabilityRegistry()
        prompt = reg.system_prompt()
        assert "No tools available" in prompt

    def test_parse_calls(self):
        reg = CapabilityRegistry()
        text = "<tool:calc>expression=2+2</tool>"
        calls = reg.parse_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "calc"

    def test_parse_calls_multiple(self):
        reg = CapabilityRegistry()
        text = "<tool:calc>expression=2+2</tool> <tool:read>path=/tmp</tool>"
        calls = reg.parse_calls(text)
        assert len(calls) == 2

    def test_execute_call(self):
        reg = CapabilityRegistry()
        reg.register(Capability(
            name="calc", description="math",
            fn=lambda expression: str(eval(expression)),
            parameters={"expression": "str"},
        ))
        call = ToolCall(tool_name="calc", arguments="expression=2+2")
        result = reg.execute_call(call)
        assert result.success is True
        assert result.output == "4"

    def test_execute_call_unknown_tool(self):
        reg = CapabilityRegistry()
        call = ToolCall(tool_name="nonexistent", arguments="")
        result = reg.execute_call(call)
        assert result.success is False

    def test_execute_all(self):
        reg = CapabilityRegistry()
        reg.register(Capability(
            name="calc", description="math",
            fn=lambda expression: str(eval(expression)),
        ))
        text = "<tool:calc>expression=3+4</tool>"
        results = reg.execute_all(text)
        assert len(results) == 1
        assert results[0].output == "7"

    def test_format_results(self):
        reg = CapabilityRegistry()
        results = [
            ToolResult(tool_name="calc", output="42", success=True),
            ToolResult(tool_name="read", output="", success=False, error="not found"),
        ]
        formatted = reg.format_results(results)
        assert "calc" in formatted
        assert "read" in formatted
        assert "42" in formatted

    def test_to_tool_registry(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="calc", description="math", fn=lambda: "ok"))
        tr = reg.to_tool_registry()
        assert isinstance(tr, ToolRegistry)
        assert tr.get("calc") is not None


# ---------------------------------------------------------------------------
# RequestContext
# ---------------------------------------------------------------------------

class TestRequestContext:
    def test_creation_defaults(self):
        ctx = RequestContext()
        assert ctx.request_id is not None
        assert ctx.user_message == ""
        assert ctx.capabilities_used == []
        assert ctx.tool_results == []
        assert ctx.max_tool_rounds == 5
        assert isinstance(ctx.working_memory, WorkingMemory)

    def test_creation_with_values(self):
        ctx = RequestContext(
            user_message="Hello",
            max_tool_rounds=3,
            metadata={"key": "val"},
        )
        assert ctx.user_message == "Hello"
        assert ctx.max_tool_rounds == 3
        assert ctx.metadata["key"] == "val"

    def test_request_id_unique(self):
        ctx1 = RequestContext()
        ctx2 = RequestContext()
        assert ctx1.request_id != ctx2.request_id

    def test_add_user_message(self):
        ctx = RequestContext(user_message="Hi there")
        ctx.add_user_message()
        entries = ctx.working_memory.entries()
        assert len(entries) == 1
        assert "Hi there" in entries[0].content

    def test_add_tool_result(self):
        ctx = RequestContext()
        result = ToolResult(tool_name="calc", output="42", success=True)
        ctx.add_tool_result(result)
        assert result in ctx.tool_results
        assert "calc" in ctx.capabilities_used
        entries = ctx.working_memory.entries()
        assert len(entries) >= 1

    def test_add_assistant_message(self):
        ctx = RequestContext()
        ctx.add_assistant_message("Hello from assistant")
        entries = ctx.working_memory.entries()
        assert len(entries) == 1
        assert "Hello from assistant" in entries[0].content

    def test_add_tool_result_failed(self):
        ctx = RequestContext()
        result = ToolResult(tool_name="calc", output="", success=False, error="bad")
        ctx.add_tool_result(result)
        assert result in ctx.tool_results
        assert "calc" in ctx.capabilities_used


# ---------------------------------------------------------------------------
# PermissionPolicy
# ---------------------------------------------------------------------------

class TestPermissionPolicy:
    def test_defaults(self):
        policy = PermissionPolicy()
        assert policy.allowed_tools is None
        assert policy.denied_tools == set()
        assert policy.require_sandbox == set()

    def test_custom_allowed(self):
        policy = PermissionPolicy(allowed_tools={"calc", "read"})
        assert policy.allowed_tools == {"calc", "read"}

    def test_custom_denied(self):
        policy = PermissionPolicy(denied_tools={"shell", "network"})
        assert policy.denied_tools == {"shell", "network"}

    def test_custom_require_sandbox(self):
        policy = PermissionPolicy(require_sandbox={"shell"})
        assert policy.require_sandbox == {"shell"}


# ---------------------------------------------------------------------------
# PermissionCheck
# ---------------------------------------------------------------------------

class TestPermissionCheck:
    def test_allowed(self):
        policy = PermissionPolicy()
        check = PermissionCheck(policy)
        ok, reason = check.is_allowed("calc")
        assert ok is True
        assert reason == "allowed"

    def test_denied(self):
        policy = PermissionPolicy(denied_tools={"shell"})
        check = PermissionCheck(policy)
        ok, reason = check.is_allowed("shell")
        assert ok is False
        assert "denied" in reason

    def test_whitelist_allowed(self):
        policy = PermissionPolicy(allowed_tools={"calc", "read"})
        check = PermissionCheck(policy)
        ok, reason = check.is_allowed("calc")
        assert ok is True

    def test_whitelist_not_allowed(self):
        policy = PermissionPolicy(allowed_tools={"calc"})
        check = PermissionCheck(policy)
        ok, reason = check.is_allowed("shell")
        assert ok is False
        assert "not in the allowed list" in reason

    def test_denied_overrides_allowed(self):
        policy = PermissionPolicy(allowed_tools={"calc", "shell"}, denied_tools={"shell"})
        check = PermissionCheck(policy)
        ok, _ = check.is_allowed("shell")
        assert ok is False

    def test_needs_sandbox_true(self):
        policy = PermissionPolicy(require_sandbox={"shell"})
        check = PermissionCheck(policy)
        assert check.needs_sandbox("shell") is True

    def test_needs_sandbox_false(self):
        policy = PermissionPolicy()
        check = PermissionCheck(policy)
        assert check.needs_sandbox("calc") is False

    def test_reason_contains_tool_name(self):
        policy = PermissionPolicy(denied_tools={"dangerous_tool"})
        check = PermissionCheck(policy)
        ok, reason = check.is_allowed("dangerous_tool")
        assert ok is False
        assert "dangerous_tool" in reason


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_success(self):
        sandbox = Sandbox(timeout_seconds=5.0)
        result = sandbox.execute(lambda **kw: "42", tool_name="calc", **{"expression": "2+2"})
        assert result.success is True
        assert result.output == "42"

    def test_returns_tool_result(self):
        sandbox = Sandbox()
        result = sandbox.execute(lambda **kw: "ok", tool_name="test")
        assert isinstance(result, ToolResult)
        assert result.tool_name == "test"

    def test_error_propagation(self):
        sandbox = Sandbox()
        result = sandbox.execute(
            lambda **kw: (_ for _ in ()).throw(ValueError("boom")),
            tool_name="bad",
        )
        assert result.success is False
        assert "boom" in result.error

    def test_timeout(self):
        sandbox = Sandbox(timeout_seconds=0.01)
        result = sandbox.execute(lambda **kw: time.sleep(5), tool_name="slow")
        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_none_output(self):
        sandbox = Sandbox()
        result = sandbox.execute(lambda **kw: None, tool_name="none_tool")
        assert result.success is True
        assert result.output == ""

    def test_numeric_output(self):
        sandbox = Sandbox()
        result = sandbox.execute(lambda **kw: 42, tool_name="num")
        assert result.success is True
        assert result.output == "42"

    def test_string_output(self):
        sandbox = Sandbox()
        result = sandbox.execute(lambda **kw: "hello", tool_name="str_tool")
        assert result.success is True
        assert result.output == "hello"

    def test_default_timeout(self):
        sandbox = Sandbox()
        assert sandbox.timeout_seconds == 30.0


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class TestPolicyEngine:
    def test_check_round_limit_not_reached(self):
        ctx = RequestContext()
        ctx.tool_results = []
        engine = PolicyEngine(max_rounds=5)
        assert engine.check_round_limit(ctx) is False

    def test_check_round_limit_reached(self):
        ctx = RequestContext()
        ctx.tool_results = [ToolResult(tool_name="x", output="y") for _ in range(5)]
        engine = PolicyEngine(max_rounds=5)
        assert engine.check_round_limit(ctx) is True

    def test_should_stop_no_tool_calls(self):
        ctx = RequestContext()
        engine = PolicyEngine(max_rounds=5)
        should_stop, reason = engine.should_stop(ctx, "Final answer: 42")
        assert should_stop is True
        assert reason == "no_tool_calls"

    def test_should_stop_round_limit(self):
        ctx = RequestContext()
        ctx.tool_results = [ToolResult(tool_name="x", output="y") for _ in range(5)]
        engine = PolicyEngine(max_rounds=5)
        should_stop, reason = engine.should_stop(ctx, "<tool:calc>expression=2+2</tool>")
        assert should_stop is True
        assert reason == "round_limit"

    def test_should_continue(self):
        ctx = RequestContext()
        engine = PolicyEngine(max_rounds=5)
        should_stop, reason = engine.should_stop(ctx, "<tool:calc>expression=2+2</tool>")
        assert should_stop is False
        assert reason == "continue"

    def test_audit_emits_log(self):
        ctx = RequestContext()
        engine = PolicyEngine(audit_log=True)
        # Should not raise
        engine.audit(ctx, "test_action", "test_detail")

    def test_audit_disabled(self):
        ctx = RequestContext()
        engine = PolicyEngine(audit_log=False)
        # Should not raise
        engine.audit(ctx, "test_action", "test_detail")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class _MockGenerator:
    """A mock generator that returns predefined responses."""

    def __init__(self, responses: list[str] | str = "Default response."):
        self._responses = responses if isinstance(responses, list) else [responses]
        self._call_count = 0

    def generate(self, prompt, *, max_new_tokens=None, temperature=None, **kw):
        if self._call_count < len(self._responses):
            result = self._responses[self._call_count]
        else:
            result = self._responses[-1]
        self._call_count += 1
        result_obj = MagicMock()
        result_obj.text = result
        result_obj.token_ids = []
        return result_obj


class TestOrchestrator:
    def test_creation_defaults(self):
        orch = Orchestrator()
        assert orch._agent is not None
        assert orch._registry is not None
        assert orch._permissions is not None
        assert orch._sandbox is not None

    def test_list_capabilities(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="calc", description="math", tags=["math"]))
        orch = Orchestrator(capability_registry=reg)
        caps = orch.list_capabilities()
        assert len(caps) == 1
        assert caps[0]["name"] == "calc"
        assert caps[0]["tags"] == ["math"]

    def test_handle_request_no_tool_calls(self):
        mock_gen = _MockGenerator("Final answer: 42")
        agent = Agent(capability_registry=CapabilityRegistry(), generator=mock_gen)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="What is 2+2?"))
        assert resp.success is True
        assert "42" in resp.text
        assert resp.tool_calls == []
        assert resp.rounds == 0

    def test_handle_request_tool_call(self):
        mock_gen = _MockGenerator([
            "<tool:calc>expression=2+2</tool>",
            "Final answer: 4. The calculation is done.",
        ])
        reg = CapabilityRegistry()
        reg.register(Capability(
            name="calc", description="math",
            fn=lambda **kw: str(eval(kw["expression"])),
        ))
        agent = Agent(capability_registry=reg, generator=mock_gen)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="What is 2+2?"))
        assert resp.success is True
        assert resp.rounds >= 1
        assert len(resp.tool_calls) >= 1
        assert len(resp.tool_results) >= 1
        assert resp.tool_results[0].output == "4"

    def test_handle_request_permission_denied(self):
        mock_gen = _MockGenerator("<tool:secret>action=read</tool>")
        reg = CapabilityRegistry()
        reg.register(Capability(name="secret", description="secret", fn=lambda **kw: "data"))
        policy = PermissionPolicy(denied_tools={"secret"})
        agent = Agent(capability_registry=reg, generator=mock_gen)
        orch = Orchestrator(agent=agent, permissions=policy)
        resp = orch.handle_request(ChatRequest(message="read secret"))
        assert resp.success is True
        assert any(not r.success for r in resp.tool_results)

    def test_handle_request_fallback_no_generator(self):
        agent = Agent(capability_registry=CapabilityRegistry(), generator=None)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="Hello"))
        assert resp.success is True
        assert "Hello" in resp.text

    def test_handle_request_fallback_with_tool_call(self):
        mock_gen = _MockGenerator("<tool:calc>expression=2+2</tool>")
        reg = CapabilityRegistry()
        reg.register(Capability(
            name="calc", description="math",
            fn=lambda **kw: str(eval(kw["expression"])),
        ))
        agent = Agent(capability_registry=reg, generator=mock_gen)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="compute <tool:calc>expression=2+2</tool>"))
        assert resp.success is True

    def test_handle_request_plain_text(self):
        mock_gen = _MockGenerator("Just a plain text response.")
        agent = Agent(capability_registry=CapabilityRegistry(), generator=mock_gen)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="Hi"))
        assert resp.success is True
        assert resp.text == "Just a plain text response."
        assert resp.tool_calls == []

    def test_handle_request_request_id(self):
        mock_gen = _MockGenerator("response")
        agent = Agent(capability_registry=CapabilityRegistry(), generator=mock_gen)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="Hi"))
        assert resp.request_id != ""

    def test_handle_request_elapsed(self):
        mock_gen = _MockGenerator("response")
        agent = Agent(capability_registry=CapabilityRegistry(), generator=mock_gen)
        orch = Orchestrator(agent=agent)
        resp = orch.handle_request(ChatRequest(message="Hi"))
        assert resp.elapsed_seconds >= 0.0

    def test_chat_request_defaults(self):
        req = ChatRequest(message="hi")
        assert req.message == "hi"
        assert req.max_rounds == 5
        assert req.metadata == {}

    def test_chat_response_to_dict(self):
        resp = ChatResponse(text="hi", success=True, rounds=1, request_id="abc")
        d = resp.to_dict()
        assert d["text"] == "hi"
        assert d["success"] is True
        assert d["rounds"] == 1
        assert d["request_id"] == "abc"

    def test_chat_response_to_dict_with_tool_calls(self):
        call = ToolCall(tool_name="calc", arguments="2+2")
        result = ToolResult(tool_name="calc", output="4", success=True)
        resp = ChatResponse(
            text="done", success=True,
            tool_calls=[call], tool_results=[result],
            rounds=1, request_id="abc",
        )
        d = resp.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["tool"] == "calc"
        assert d["tool_results"][0]["output"] == "4"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    def test_creation_no_generator(self):
        reg = CapabilityRegistry()
        agent = Agent(capability_registry=reg, generator=None)
        assert agent.generator is None
        assert agent.capability_registry is reg

    def test_from_config_with_builtin_tools(self):
        agent = Agent.from_config()
        caps = agent.capability_registry.list_capabilities()
        names = [c.name for c in caps]
        assert "calculator" in names
        assert "read_file" in names

    def test_from_config_with_custom_registry(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="custom", description="d"))
        agent = Agent.from_config(capability_registry=reg)
        caps = agent.capability_registry.list_capabilities()
        names = [c.name for c in caps]
        assert "custom" in names
        assert "calculator" not in names

    def test_use_tool_success(self):
        agent = Agent.from_config()
        result = agent.use_tool("calculator", {"expression": "2+2"})
        assert result.success is True
        assert result.output == "4"

    def test_use_tool_unknown(self):
        agent = Agent.from_config()
        result = agent.use_tool("nonexistent", {})
        assert result.success is False
        assert "Unknown" in result.error

    def test_use_tool_no_fn(self):
        reg = CapabilityRegistry()
        reg.register(Capability(name="empty", description="d"))
        agent = Agent(capability_registry=reg)
        result = agent.use_tool("empty", {})
        assert result.success is False

    def test_chat_no_generator(self):
        agent = Agent.from_config()
        ctx = RequestContext(user_message="Hi")
        result = agent.chat("Hi", ctx)
        assert "Generator not available" in result


# ---------------------------------------------------------------------------
# Tools (calculator, read_file)
# ---------------------------------------------------------------------------

class TestTools:
    def test_calculator_addition(self):
        assert calculator("2+2") == "4"

    def test_calculator_subtraction(self):
        assert calculator("10-3") == "7"

    def test_calculator_multiplication(self):
        assert calculator("6*7") == "42"

    def test_calculator_division(self):
        assert calculator("10/4") == "2.5"

    def test_calculator_power(self):
        assert calculator("2**10") == "1024"

    def test_calculator_parentheses(self):
        assert calculator("(3+4)*2") == "14"

    def test_calculator_negative(self):
        assert calculator("-5") == "-5"

    def test_calculator_float_result(self):
        result = calculator("10/3")
        assert float(result) > 3.3
        assert float(result) < 3.4

    def test_calculator_empty_raises(self):
        try:
            calculator("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_read_file_success(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            content = read_file(path)
            assert content == "hello world"
        finally:
            os.unlink(path)

    def test_read_file_not_found(self):
        try:
            read_file("/nonexistent/file/path")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_register_builtin_tools(self):
        reg = CapabilityRegistry()
        register_builtin_tools(reg)
        caps = reg.list_capabilities()
        names = [c.name for c in caps]
        assert "calculator" in names
        assert "read_file" in names


# ---------------------------------------------------------------------------
# WorkflowStep, Workflow, WorkflowResult, StepResult
# ---------------------------------------------------------------------------

class TestWorkflowStep:
    def test_step_render(self):
        step = WorkflowStep(
            name="calc", capability_name="calculator",
            input_template="compute {user_input}",
        )
        result = step.render("What is 2+2?")
        assert "What is 2+2?" in result

    def test_step_render_defaults(self):
        step = WorkflowStep(name="s", capability_name="calc")
        result = step.render("input", "prev")
        assert result == "prev"

    def test_step_defaults(self):
        step = WorkflowStep(name="s", capability_name="calc")
        assert step.input_template == "{prev_output}"
        assert step.description == ""


class TestWorkflow:
    def test_workflow_creation(self):
        wf = Workflow(name="test_wf", description="a test")
        assert wf.name == "test_wf"
        assert wf.description == "a test"
        assert wf.steps == []

    def test_workflow_with_steps(self):
        steps = [
            WorkflowStep(name="s1", capability_name="calc"),
            WorkflowStep(name="s2", capability_name="read"),
        ]
        wf = Workflow(name="wf", steps=steps)
        assert len(wf.steps) == 2

    def test_workflow_step_result(self):
        step = WorkflowStep(name="s", capability_name="calc")
        result = StepResult(
            step=step, input_text="in", output_text="out",
            success=True, rounds=1,
        )
        assert result.output_text == "out"
        assert result.success is True
        assert result.rounds == 1

    def test_workflow_result_to_dict(self):
        step = WorkflowStep(name="s", capability_name="calc")
        sr = StepResult(step=step, input_text="in", output_text="out")
        wf = Workflow(name="wf", steps=[step])
        result = WorkflowResult(
            workflow=wf, input_text="test", output_text="done",
            step_results=[sr],
        )
        d = result.to_dict()
        assert d["workflow"] == "wf"
        assert d["input"] == "test"
        assert d["output"] == "done"
        assert len(d["steps"]) == 1

    def test_workflow_result_all_succeeded(self):
        step = WorkflowStep(name="s", capability_name="calc")
        sr = StepResult(step=step, input_text="in", output_text="out", success=True)
        wf = Workflow(name="wf", steps=[step])
        result = WorkflowResult(workflow=wf, input_text="x", output_text="y", step_results=[sr])
        assert result.all_succeeded is True

    def test_workflow_result_all_failed(self):
        step = WorkflowStep(name="s", capability_name="calc")
        sr = StepResult(step=step, input_text="in", output_text="out", success=False)
        wf = Workflow(name="wf", steps=[step])
        result = WorkflowResult(workflow=wf, input_text="x", output_text="y", step_results=[sr])
        assert result.all_succeeded is False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestRuntimeImports:
    def test_all_imports(self):
        """All public symbols are importable from runtime package."""
        import runtime
        assert hasattr(runtime, "Capability")
        assert hasattr(runtime, "CapabilityRegistry")
        assert hasattr(runtime, "ChatRequest")
        assert hasattr(runtime, "ChatResponse")
        assert hasattr(runtime, "Orchestrator")
        assert hasattr(runtime, "PermissionCheck")
        assert hasattr(runtime, "PermissionPolicy")
        assert hasattr(runtime, "PolicyEngine")
        assert hasattr(runtime, "RequestContext")
        assert hasattr(runtime, "Sandbox")
        assert hasattr(runtime, "IntelligenceHandler")
