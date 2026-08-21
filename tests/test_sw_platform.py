"""Tests for the sw_platform/ package (Layers 2, 3, and 4).

Covers:
    - Harness agent (PydanticAgentHarness, HarnessConfig, AgentResponse)
    - Tool providers (CodeExecution, Database, Filesystem, Git, WebAutomation)
    - Coder module (CoderProvider, RepoContext, DockerSandbox, StructuredOutput)
    - pydantic_ai integration (create_harness_agent)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sw_platform.coder import (
    CodeExplanation,
    CodePatch,
    CoderProvider,
    DockerSandbox,
    RepoContext,
    StructuredOutput,
)
from sw_platform.harness.agent import (
    AgentResponse,
    HarnessConfig,
    PydanticAgentHarness,
    ToolCallRecord,
    create_harness_agent,
)
from sw_platform.harness.core import ExecutionResult, ToolSpec
from sw_platform.tools.code_execution import CodeExecutionProvider
from sw_platform.tools.database import DatabaseProvider
from sw_platform.tools.filesystem import FilesystemProvider
from sw_platform.tools.git import GitProvider
from sw_platform.tools.web_automation import WebAutomationProvider

# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_creation_success(self):
        result = ExecutionResult(
            tool_name="test",
            success=True,
            output="hello",
            elapsed_seconds=0.5,
        )
        assert result.tool_name == "test"
        assert result.success is True
        assert result.output == "hello"
        assert result.error == ""
        assert result.elapsed_seconds == 0.5

    def test_creation_failure(self):
        result = ExecutionResult(
            tool_name="test",
            success=False,
            error="something went wrong",
        )
        assert result.success is False
        assert result.error == "something went wrong"
        assert result.output == ""

    def test_defaults(self):
        result = ExecutionResult(tool_name="t", success=False)
        assert result.output == ""
        assert result.error == ""
        assert result.elapsed_seconds == 0.0
        assert result.metadata == {}

    def test_to_dict(self):
        result = ExecutionResult(
            tool_name="test",
            success=True,
            output="42",
            error="",
            elapsed_seconds=1.5,
        )
        d = result.to_dict()
        assert d["tool_name"] == "test"
        assert d["success"] is True
        assert d["output"] == "42"
        assert d["error"] == ""
        assert d["elapsed_seconds"] == 1.5


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------

class TestToolSpec:
    def test_creation(self):
        spec = ToolSpec(
            name="run_python",
            description="Execute Python code",
            parameters={"code": "str"},
            tags=["execution", "python"],
            risk_level="medium",
            permission_required="execute",
        )
        assert spec.name == "run_python"
        assert spec.description == "Execute Python code"
        assert spec.parameters == {"code": "str"}
        assert spec.tags == ["execution", "python"]
        assert spec.risk_level == "medium"
        assert spec.permission_required == "execute"

    def test_defaults(self):
        spec = ToolSpec(
            name="test",
            description="Test tool",
        )
        assert spec.parameters == {}
        assert spec.tags == []
        assert spec.risk_level == "low"
        assert spec.permission_required == "read"


# ---------------------------------------------------------------------------
# CodeExecutionProvider
# ---------------------------------------------------------------------------

class TestCodeExecutionProvider:
    def test_get_tools(self):
        provider = CodeExecutionProvider()
        tools = provider.get_tools()
        names = [t.name for t in tools]
        assert "run_python" in names
        assert "python_ast" in names

    def test_run_python_success(self):
        provider = CodeExecutionProvider()
        result = provider.execute("run_python", code="print('hello world')")
        assert result.success is True
        assert "hello world" in result.output

    def test_run_python_calculation(self):
        provider = CodeExecutionProvider()
        result = provider.execute("run_python", code="print(2 + 3)")
        assert result.success is True
        assert "5" in result.output

    def test_run_python_error(self):
        provider = CodeExecutionProvider()
        result = provider.execute("run_python", code="raise ValueError('test error')")
        assert result.success is False
        assert "test error" in result.error

    def test_run_python_syntax_error(self):
        provider = CodeExecutionProvider()
        result = provider.execute("run_python", code="def broken(:")
        assert result.success is False
        assert "SyntaxError" in result.error or "syntax" in result.error.lower()

    def test_python_ast_success(self):
        provider = CodeExecutionProvider()
        result = provider.execute("python_ast", expression="42")
        assert result.success is True
        assert "42" in result.output

    def test_python_ast_safe_eval(self):
        provider = CodeExecutionProvider()
        result = provider.execute("python_ast", expression="{'key': 'value'}")
        assert result.success is True
        assert "key" in result.output

    def test_python_ast_error(self):
        provider = CodeExecutionProvider()
        result = provider.execute("python_ast", expression="__import__('os')")
        assert result.success is False

    def test_unknown_tool(self):
        provider = CodeExecutionProvider()
        result = provider.execute("nonexistent_tool")
        assert result.success is False
        assert "Unknown" in result.error

    def test_run_python_variables(self):
        provider = CodeExecutionProvider()
        code = "x = 10\ny = 20\nprint(x + y)"
        result = provider.execute("run_python", code=code)
        assert result.success is True
        assert "30" in result.output


# ---------------------------------------------------------------------------
# FilesystemProvider
# ---------------------------------------------------------------------------

class TestFilesystemProvider:
    def test_get_tools(self):
        provider = FilesystemProvider()
        tools = provider.get_tools()
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "write_file" in names
        assert "list_directory" in names
        assert "move_file" in names
        assert "delete_file" in names

    def test_list_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FilesystemProvider(allowed_paths=[tmpdir])
            result = provider.execute("list_directory", path=tmpdir)
            assert result.success is True
            assert isinstance(result.output, str)

    def test_write_and_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FilesystemProvider(allowed_paths=[tmpdir])
            test_file = os.path.join(tmpdir, "test.txt")

            result = provider.execute("write_file", path=test_file, content="hello file")
            assert result.success is True

            result = provider.execute("read_file", path=test_file)
            assert result.success is True
            assert "hello file" in result.output

    def test_read_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FilesystemProvider(allowed_paths=[tmpdir])
            result = provider.execute("read_file", path="nonexistent.txt")
            assert result.success is False

    def test_path_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FilesystemProvider(allowed_paths=[tmpdir])
            result = provider.execute("read_file", path="/etc/passwd")
            assert result.success is False
            assert "not in allowed list" in result.error.lower() or "not allowed" in result.error.lower()


# ---------------------------------------------------------------------------
# DatabaseProvider
# ---------------------------------------------------------------------------

class TestDatabaseProvider:
    def test_get_tools(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            provider = DatabaseProvider(database_path=f.name)
            tools = provider.get_tools()
            names = [t.name for t in tools]
            assert "sql_query" in names
            assert "sql_list_tables" in names
            assert "sql_schema" in names
            assert "sql_explain" in names

    def test_sql_query(self):
        import sqlite3
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'hello')")
            conn.execute("INSERT INTO test VALUES (2, 'world')")
            conn.commit()
            conn.close()

            provider = DatabaseProvider(database_path=db_path)
            result = provider.execute("sql_query", query="SELECT * FROM test")
            assert result.success is True
            assert "hello" in result.output.lower() or "1" in result.output
        finally:
            os.unlink(db_path)

    def test_sql_query_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            provider = DatabaseProvider(database_path=db_path)
            result = provider.execute("sql_query", query="SELECT * FROM nonexistent_table")
            assert result.success is False
            assert result.tool_name == "sql_query"
        finally:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass


# ---------------------------------------------------------------------------
# GitProvider
# ---------------------------------------------------------------------------

class TestGitProvider:
    def test_get_tools(self):
        provider = GitProvider()
        tools = provider.get_tools()
        names = [t.name for t in tools]
        assert "git_status" in names
        assert "git_diff" in names
        assert "git_log" in names
        assert "git_add" in names
        assert "git_commit" in names
        assert "git_blame" in names
        assert "git_clone" in names

    def test_git_status(self):
        provider = GitProvider(repo_path=os.getcwd())
        result = provider.execute("git_status")
        assert result.tool_name == "git_status"
        assert result.elapsed_seconds > 0

    def test_git_log(self):
        provider = GitProvider(repo_path=os.getcwd())
        result = provider.execute("git_log")
        assert result.tool_name == "git_log"

    def test_git_diff_unstaged(self):
        provider = GitProvider(repo_path=os.getcwd())
        result = provider.execute("git_diff", target="unstaged")
        assert result.tool_name == "git_diff"

    def test_unknown_tool(self):
        provider = GitProvider()
        result = provider.execute("nonexistent")
        assert result.success is False
        assert "Unknown" in result.error

    def test_git_blame(self):
        provider = GitProvider(repo_path=os.getcwd())
        result = provider.execute("git_blame", path="pyproject.toml")
        assert result.tool_name == "git_blame"


# ---------------------------------------------------------------------------
# WebAutomationProvider
# ---------------------------------------------------------------------------

class TestWebAutomationProvider:
    def test_get_tools(self):
        provider = WebAutomationProvider()
        tools = provider.get_tools()
        names = [t.name for t in tools]
        assert "web_fetch" in names
        assert "web_scrape" in names
        assert "web_form_fill" in names

    def test_web_fetch_no_network(self):
        """Test that web_fetch handles network errors gracefully."""
        provider = WebAutomationProvider()
        result = provider.execute("web_fetch", url="http://nonexistent.invalid.domain.xyz")
        assert result.tool_name == "web_fetch"
        # It should fail gracefully (network error), not crash
        assert result.success is False or isinstance(result.output, str)

    def test_unknown_tool(self):
        provider = WebAutomationProvider()
        result = provider.execute("nonexistent")
        assert result.success is False
        assert "Unknown" in result.error


# ---------------------------------------------------------------------------
# RepoContext
# ---------------------------------------------------------------------------

class TestRepoContext:
    def test_init(self):
        ctx = RepoContext(repo_root=os.getcwd())
        assert ctx.repo_root == os.getcwd()
        assert ctx.file_count > 0

    def test_search(self):
        ctx = RepoContext(repo_root=os.getcwd())
        results = ctx.search("pyproject")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_no_results(self):
        ctx = RepoContext(repo_root=os.getcwd())
        results = ctx.search("zzz_nonexistent_file_xyz")
        assert len(results) == 0

    def test_search_limit(self):
        ctx = RepoContext(repo_root=os.getcwd())
        results = ctx.search("test", limit=2)
        assert len(results) <= 2

    def test_get_language_summary(self):
        ctx = RepoContext(repo_root=os.getcwd())
        summary = ctx.get_language_summary()
        assert isinstance(summary, dict)
        assert "python" in summary

    def test_get_repo_hash(self):
        ctx = RepoContext(repo_root=os.getcwd())
        h = ctx.get_repo_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_read_file(self):
        ctx = RepoContext(repo_root=os.getcwd())
        content = ctx.read_file("pyproject.toml")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_read_file_not_found(self):
        ctx = RepoContext(repo_root=os.getcwd())
        content = ctx.read_file("nonexistent_file.txt")
        assert "Error" in content

    def test_refresh(self):
        ctx = RepoContext(repo_root=os.getcwd())
        ctx.refresh()
        assert ctx.file_count > 0

    def test_get_file(self):
        ctx = RepoContext(repo_root=os.getcwd())
        entry = ctx.get_file("pyproject.toml")
        assert entry is not None
        assert entry.language == "toml"


# ---------------------------------------------------------------------------
# StructuredOutput
# ---------------------------------------------------------------------------

class TestStructuredOutput:
    def test_validate_success(self):
        data = {"file_path": "a.py", "old_content": "x", "new_content": "y"}
        success, model, error = StructuredOutput.validate(data, CodePatch)
        assert success is True
        assert error == ""
        assert isinstance(model, CodePatch)
        assert model.file_path == "a.py"

    def test_validate_failure(self):
        data = {"file_path": "a.py"}  # missing required fields
        success, _, error = StructuredOutput.validate(data, CodePatch)
        assert success is False
        assert error != ""

    def test_validate_code_explanation(self):
        data = {
            "file_path": "test.py",
            "language": "python",
            "summary": "A test file",
        }
        success, model, error = StructuredOutput.validate(data, CodeExplanation)
        assert success is True
        assert isinstance(model, CodeExplanation)
        assert model.file_path == "test.py"
        assert model.language == "python"

    def test_validate_code_explanation_with_defaults(self):
        data = {
            "file_path": "test.py",
            "language": "python",
            "summary": "A test file",
            "key_functions": ["main"],
            "complexity_rating": "high",
        }
        success, model, _ = StructuredOutput.validate(data, CodeExplanation)
        assert success is True
        assert model.key_functions == ["main"]
        assert model.complexity_rating == "high"


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------

class TestDockerSandbox:
    def test_creation(self):
        sandbox = DockerSandbox()
        assert sandbox is not None

    def test_docker_check(self):
        sandbox = DockerSandbox()
        _ = sandbox.is_docker_available

    def test_run_local_fallback(self):
        sandbox = DockerSandbox(fallback_to_local=True)
        result = sandbox.run("print('hello from sandbox')")
        assert result.tool_name == "docker_sandbox"
        assert result.success is True
        assert "hello from sandbox" in result.output

    def test_run_local_error(self):
        sandbox = DockerSandbox(fallback_to_local=True)
        result = sandbox.run("raise RuntimeError('test')")
        assert result.success is False
        assert "test" in result.error

    def test_run_local_calculation(self):
        sandbox = DockerSandbox(fallback_to_local=True)
        result = sandbox.run("print(2 + 3)")
        assert result.success is True
        assert "5" in result.output

    def test_run_local_timeout(self):
        sandbox = DockerSandbox(fallback_to_local=True, timeout=1.0)
        result = sandbox.run("import time; time.sleep(10)")
        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_unsupported_language(self):
        sandbox = DockerSandbox(fallback_to_local=True)
        result = sandbox.run("print('hi')", language="javascript")
        assert result.success is False
        assert "not supported" in result.error


# ---------------------------------------------------------------------------
# CoderProvider
# ---------------------------------------------------------------------------

class TestCoderProvider:
    def test_get_tools(self):
        provider = CoderProvider()
        tools = provider.get_tools()
        names = [t.name for t in tools]
        assert "run_python_code" in names
        assert "repo_search" in names
        assert "repo_read" in names
        assert "repo_summary" in names
        assert "validate_json" in names
        assert "run_in_sandbox" in names

    def test_run_python_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("run_python_code", code="print(42)")
            assert result.success is True
            assert "42" in result.output

    def test_repo_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("repo_search", query="python")
            assert result.tool_name == "repo_search"

    def test_repo_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("print('hello')")

            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("repo_read", path="test.py")
            assert result.success is True
            assert "hello" in result.output

    def test_repo_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("repo_summary")
            assert result.tool_name == "repo_summary"

    def test_validate_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("validate_json", json_str='{"key": "value"}')
            assert result.success is True
            assert "Valid JSON" in result.output

    def test_validate_json_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("validate_json", json_str="{invalid json}")
            assert result.success is False
            assert "Invalid JSON" in result.error

    def test_validate_json_missing_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute(
                "validate_json", json_str='{"a": 1}', required_keys=["missing_key"]
            )
            assert result.success is False
            assert "Missing" in result.error

    def test_run_in_sandbox(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("run_in_sandbox", code="print('sandboxed')")
            assert result.tool_name == "run_in_sandbox"

    def test_run_python_code_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("run_python_code", code="raise ValueError('test')")
            assert result.success is False
            assert "test" in result.error

    def test_unknown_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CoderProvider(repo_root=tmpdir)
            result = provider.execute("nonexistent")
            assert result.success is False
            assert "Unknown" in result.error


# ---------------------------------------------------------------------------
# HarnessConfig
# ---------------------------------------------------------------------------

class TestHarnessConfig:
    def test_defaults(self):
        config = HarnessConfig()
        assert config.model == "openai:gpt-4o"
        assert config.max_rounds == 5
        assert config.permission_level == "read"
        assert config.sandbox_enabled is True
        assert config.read_only_mode is True

    def test_custom(self):
        config = HarnessConfig(
            model="google:gemma-2b",
            max_rounds=10,
            permission_level="write",
            read_only_mode=False,
        )
        assert config.model == "google:gemma-2b"
        assert config.max_rounds == 10
        assert config.permission_level == "write"
        assert config.read_only_mode is False

    def test_system_prompt(self):
        config = HarnessConfig()
        assert "Silverwing" in config.system_prompt
        assert "autonomous" in config.system_prompt


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

class TestAgentResponse:
    def test_creation(self):
        resp = AgentResponse(text="Hello", success=True)
        assert resp.text == "Hello"
        assert resp.success is True
        assert resp.error == ""
        assert resp.tool_calls == []
        assert resp.elapsed_seconds == 0.0

    def test_to_dict(self):
        resp = AgentResponse(
            text="Hello", success=True, elapsed_seconds=1.23,
        )
        d = resp.to_dict()
        assert d["text"] == "Hello"
        assert d["success"] is True
        assert d["elapsed_seconds"] == 1.23

    def test_with_tool_calls(self):
        resp = AgentResponse(text="Used a tool", success=True)
        assert len(resp.tool_calls) == 0


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------

class TestToolCallRecord:
    def test_to_dict(self):
        result = ExecutionResult(tool_name="test", success=True, output="42")
        record = ToolCallRecord(
            tool_name="test",
            arguments={"code": "print(42)"},
            result=result,
            timestamp=time.time(),
        )
        d = record.to_dict()
        assert d["tool_name"] == "test"
        assert d["arguments"] == {"code": "print(42)"}
        assert d["result"]["success"] is True
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# PydanticAgentHarness
# ---------------------------------------------------------------------------

class TestPydanticAgentHarness:
    def test_creation_defaults(self):
        harness = PydanticAgentHarness()
        assert harness.config.model == "openai:gpt-4o"
        assert harness.sandbox is not None

    def test_creation_custom_config(self):
        config = HarnessConfig(max_rounds=3, permission_level="write")
        harness = PydanticAgentHarness(config)
        assert harness.config.max_rounds == 3
        assert harness.config.permission_level == "write"

    def test_tools_registered(self):
        harness = PydanticAgentHarness()
        tools = harness.tools
        assert len(tools) > 0
        names = [t.name for t in tools]
        assert "run_python" in names
        assert "read_file" in names
        assert "git_status" in names

    def test_get_tool(self):
        harness = PydanticAgentHarness()
        tool = harness.get_tool("run_python")
        assert tool is not None
        assert tool.name == "run_python"

    def test_get_tool_not_found(self):
        harness = PydanticAgentHarness()
        assert harness.get_tool("nonexistent") is None

    def test_register_provider(self):
        harness = PydanticAgentHarness()
        initial_count = len(harness.tools)
        provider = CodeExecutionProvider()
        harness.register_provider(provider)
        new_count = len(harness.tools)
        assert new_count >= initial_count

    def test_build_system_prompt(self):
        harness = PydanticAgentHarness()
        prompt = harness._build_system_prompt()
        assert "Silverwing" in prompt
        assert "Available tools" in prompt

    def test_run_fallback_no_llm(self):
        harness = PydanticAgentHarness()
        response = harness.run("Hello")
        assert response is not None
        assert isinstance(response.text, str)
        assert response.success is True

    def test_run_fallback_list_files(self):
        harness = PydanticAgentHarness()
        response = harness.run("list files")
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_run_fallback_calculation(self):
        harness = PydanticAgentHarness()
        response = harness.run("run_python print(2 + 2)")
        assert "4" in response.text

    def test_run_resets_history(self):
        harness = PydanticAgentHarness()
        harness.run("hello", reset_history=True)
        assert len(harness._conversation_history) == 2

    def test_audit_log_populated(self):
        harness = PydanticAgentHarness()
        harness.run("hello")
        assert len(harness.audit_log) > 0

    def test_conversation_history(self):
        harness = PydanticAgentHarness()
        harness.run("test message")
        assert len(harness.conversation_history) >= 1

    def test_run_git_command(self):
        harness = PydanticAgentHarness()
        response = harness.run("git status")
        assert isinstance(response.text, str)

    def test_run_async(self):
        harness = PydanticAgentHarness()
        response = harness.run_async("hello")
        assert response is not None
        assert isinstance(response, AgentResponse)


# ---------------------------------------------------------------------------
# create_harness_agent factory
# ---------------------------------------------------------------------------

class TestCreateHarnessAgent:
    def test_default_creation(self):
        harness = create_harness_agent()
        assert harness is not None
        assert isinstance(harness, PydanticAgentHarness)
        assert len(harness.tools) > 0

    def test_custom_model(self):
        harness = create_harness_agent(
            model="google:gemma-2b", max_rounds=3
        )
        assert harness.config.model == "google:gemma-2b"
        assert harness.config.max_rounds == 3

    def test_run(self):
        harness = create_harness_agent(repo_root=os.getcwd())
        response = harness.run("What is 2+2?")
        assert isinstance(response, AgentResponse)
        assert response.success is True

    def test_with_database(self):
        import sqlite3
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            conn.close()

            harness = create_harness_agent(
                database_path=db_path, repo_root=os.getcwd()
            )
            assert harness is not None
            assert len(harness.tools) > 14
        finally:
            os.unlink(db_path)

    def test_allowed_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = create_harness_agent(
                allowed_paths=[tmpdir],
                repo_root=os.getcwd(),
            )
            assert harness is not None
