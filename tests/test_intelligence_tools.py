"""Tests for M15.6: tool-use protocol."""

from __future__ import annotations

from intelligence.tools import Tool, ToolCall, ToolExecutor, ToolRegistry, ToolResult


def test_tool_registry_register_and_get():
    registry = ToolRegistry()
    tool = Tool(name="calc", description="Calculate", fn=lambda x="0": str(int(x) + 1))
    registry.register(tool)
    assert registry.get("calc") is tool
    assert registry.get("missing") is None


def test_tool_registry_list_tools():
    registry = ToolRegistry()
    registry.register(Tool(name="a", description="tool a"))
    registry.register(Tool(name="b", description="tool b"))
    tools = registry.list_tools()
    assert len(tools) == 2


def test_tool_registry_system_prompt():
    registry = ToolRegistry()
    registry.register(Tool(name="calc", description="Calculate stuff", parameters={"expr": "math expr"}))
    prompt = registry.system_prompt()
    assert "calc" in prompt
    assert "Calculate stuff" in prompt
    assert "expr" in prompt


def test_tool_registry_empty_system_prompt():
    registry = ToolRegistry()
    prompt = registry.system_prompt()
    assert "No tools" in prompt


def test_tool_registry_parse_calls():
    registry = ToolRegistry()
    registry.register(Tool(name="calc", description="calc"))
    calls = registry.parse_calls("Let me calculate: <tool:calc>expression=2+2</tool>")
    assert len(calls) == 1
    assert calls[0].tool_name == "calc"
    assert "expression=2+2" in calls[0].arguments


def test_tool_registry_parse_no_calls():
    registry = ToolRegistry()
    calls = registry.parse_calls("no tools here")
    assert len(calls) == 0


def test_tool_registry_parse_multiple_calls():
    registry = ToolRegistry()
    registry.register(Tool(name="a", description="a"))
    registry.register(Tool(name="b", description="b"))
    text = "<tool:a>x=1</tool> and <tool:b>y=2</tool>"
    calls = registry.parse_calls(text)
    assert len(calls) == 2


def test_tool_call_args_dict():
    call = ToolCall(tool_name="calc", arguments="expression=2+2, mode=fast")
    d = call.args_dict
    assert d["expression"] == "2+2"
    assert d["mode"] == "fast"


def test_tool_call_args_dict_single():
    call = ToolCall(tool_name="calc", arguments="hello")
    d = call.args_dict
    assert d["input"] == "hello"


def test_tool_executor_success():
    registry = ToolRegistry()
    registry.register(Tool(
        name="calc",
        description="calc",
        parameters={"x": "number"},
        fn=lambda x="0": str(int(x) * 2),
    ))
    executor = ToolExecutor(registry)
    result = executor.execute_call(ToolCall("calc", "x=5"))
    assert result.success
    assert result.output == "10"


def test_executor_unknown_tool():
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    result = executor.execute_call(ToolCall("nonexistent", ""))
    assert not result.success
    assert "Unknown" in result.error


def test_executor_tool_error():
    registry = ToolRegistry()
    registry.register(Tool(name="fail", description="fail", fn=lambda: 1 / 0))
    executor = ToolExecutor(registry)
    result = executor.execute_call(ToolCall("fail", ""))
    assert not result.success
    assert "division" in result.error.lower() or "zero" in result.error.lower()


def test_executor_tool_no_fn():
    registry = ToolRegistry()
    registry.register(Tool(name="nofn", description="no fn"))
    executor = ToolExecutor(registry)
    result = executor.execute_call(ToolCall("nofn", ""))
    assert not result.success
    assert "no implementation" in result.error


def test_executor_execute_all():
    registry = ToolRegistry()
    registry.register(Tool(name="calc", description="calc", fn=lambda x="0": x))
    executor = ToolExecutor(registry)
    results = executor.execute_all("result: <tool:calc>x=hello</tool>")
    assert len(results) == 1
    assert results[0].output == "hello"


def test_executor_format_results():
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    results = [
        ToolResult(tool_name="a", output="ok", success=True),
        ToolResult(tool_name="b", output="", success=False, error="bad"),
    ]
    text = executor.format_results(results)
    assert "Tool a output: ok" in text
    assert "Tool b error: bad" in text


def test_tool_to_prompt():
    tool = Tool(name="calc", description="Calculate", parameters={"x": "number"})
    prompt = tool.to_prompt()
    assert "calc" in prompt
    assert "Calculate" in prompt
    assert "x: number" in prompt


def test_tool_no_params_to_prompt():
    tool = Tool(name="noop", description="Does nothing")
    prompt = tool.to_prompt()
    assert "none" in prompt


def test_tool_result_success():
    r = ToolResult(tool_name="t", output="ok", success=True)
    assert r.success
    r2 = ToolResult(tool_name="t", output="", success=False, error="fail")
    assert not r2.success
