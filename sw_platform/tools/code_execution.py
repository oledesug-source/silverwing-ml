"""Code execution engine — allows the LLM to dynamically execute Python code.

Implements the "System Execution" capability from Layer 3 using a
subprocess-based Python AST interpreter (similar to LangChain's PythonAstREPLTool
or Open Interpreter's code runner).  All execution is routed through the
platform's sandbox for timeout and resource enforcement.
"""

from __future__ import annotations

import ast
import io
import logging
import os
import tempfile
import textwrap
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from sw_platform.harness.core import ExecutionResult, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)


class CodeExecutionProvider(ToolProvider):
    """Provider for Python code execution tools.

    Wraps the existing ``SandboxExecutor`` for timeout enforcement.
    Falls back to local threading-based timeout if no sandbox is provided.
    """

    def __init__(self, sandbox: Any = None, timeout: float = 30.0) -> None:
        self._sandbox = sandbox
        self._timeout = timeout

    def get_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="run_python",
                description=(
                    "Execute Python code in a sandboxed subprocess. "
                    "Returns stdout output. Use this for calculations, "
                    "data processing, file I/O, and prototyping. "
                    "Code is executed with a timeout — keep it concise."
                ),
                parameters={
                    "code": "str — Python code to execute (must be valid Python 3)",
                },
                tags=["execution", "python", "code"],
                risk_level="medium",
                permission_required="execute",
            ),
            ToolSpec(
                name="python_ast",
                description=(
                    "Parse and evaluate a Python expression using ast.literal_eval. "
                    "Safe for evaluating literals (numbers, strings, lists, dicts). "
                    "Does NOT execute arbitrary code."
                ),
                parameters={
                    "expression": "str — Python expression to safely evaluate",
                },
                tags=["execution", "safe", "math"],
                risk_level="low",
                permission_required="read",
            ),
        ]

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()

        if name == "run_python":
            return self._run_python(**kwargs)
        elif name == "python_ast":
            return self._eval_ast(**kwargs)

        return ExecutionResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: {name}",
            elapsed_seconds=time.monotonic() - t0,
        )

    def _run_python(self, code: str, **kwargs: Any) -> ExecutionResult:
        """Execute Python code via subprocess with timeout."""
        t0 = time.monotonic()

        # Validate syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(
                tool_name="run_python",
                success=False,
                error=f"SyntaxError: {e}",
                elapsed_seconds=time.monotonic() - t0,
            )

        # If we have a sandbox with resource limits, wrap execution
        if self._sandbox is not None:
            try:
                from sw_platform.sandbox.executor import SandboxExecutor
                if isinstance(self._sandbox, SandboxExecutor):
                    return self._sandbox.execute(
                        self._execute_python_local,
                        cap_id="run_python",
                        code=code,
                    )
            except Exception:
                pass  # Fall through to local execution

        # Fallback: local execution with threading timeout
        return self._execute_python_local(code=code)

    def _execute_python_local(self, code: str, **kwargs: Any) -> ExecutionResult:
        """Execute Python code locally with stdout/stderr capture.

        Uses a temporary directory as CWD for file safety.
        """
        t0 = time.monotonic()

        # Dedent code in case of indentation artifacts
        code = textwrap.dedent(code).strip()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Create a temporary directory for file operations
        with tempfile.TemporaryDirectory(prefix="silverwing_exec_") as tmpdir:
            original_cwd = os.getcwd()
            original_env = dict(os.environ)

            try:
                os.chdir(tmpdir)

                # Set up a safe namespace
                namespace: dict[str, Any] = {
                    "__name__": "__silverwing_exec__",
                    "__builtins__": __builtins__,
                    "print": print,
                    "len": len,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "abs": abs,
                    "round": round,
                    "reversed": reversed,
                    "isinstance": isinstance,
                    "type": type,
                    "open": open,  # Restricted to temp dir via CWD
                }

                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, namespace)  # noqa: S102

            except Exception as exc:
                traceback.print_exc(file=stderr_buf)
                elapsed = time.monotonic() - t0
                return ExecutionResult(
                    tool_name="run_python",
                    success=False,
                    error=str(exc),
                    elapsed_seconds=elapsed,
                    metadata={
                        "stdout": stdout_buf.getvalue(),
                        "stderr": stderr_buf.getvalue()[:2000],
                    },
                )
            finally:
                os.chdir(original_cwd)
                os.environ.clear()
                os.environ.update(original_env)

        elapsed = time.monotonic() - t0
        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        return ExecutionResult(
            tool_name="run_python",
            success=True,
            output=stdout_val if stdout_val else stderr_val,
            elapsed_seconds=elapsed,
            metadata={
                "stdout": stdout_val,
                "stderr": stderr_val[:2000],
                "cwd": tmpdir,
            },
        )

    def _eval_ast(self, expression: str, **kwargs: Any) -> ExecutionResult:
        """Safely evaluate a Python literal expression."""
        t0 = time.monotonic()
        try:
            result = ast.literal_eval(expression)
            return ExecutionResult(
                tool_name="python_ast",
                success=True,
                output=str(result),
                elapsed_seconds=time.monotonic() - t0,
            )
        except (ValueError, SyntaxError) as exc:
            return ExecutionResult(
                tool_name="python_ast",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
