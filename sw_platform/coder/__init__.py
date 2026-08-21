"""Coder module — software development capabilities for the agent.

Provides:
    - Local code interpreter (executes code snippets in-process)
    - Repository context provider (multi-file awareness for code understanding)
    - Docker sandbox adapter (E2B-compatible interface, with local fallback)
    - Structured output validation (pydantic model-based)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
import textwrap
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sw_platform.harness.core import ExecutionResult, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output models (PydanticAI/CrewAI-style)
# ---------------------------------------------------------------------------

class CodePatch(BaseModel):
    """A structured code patch proposal."""

    file_path: str = Field(..., description="Path to the file to modify")
    old_content: str = Field(..., description="Existing content to replace")
    new_content: str = Field(..., description="Replacement content")
    explanation: str = Field(default="", description="Why this change is needed")


class CodeExplanation(BaseModel):
    """Structured explanation of code."""

    file_path: str
    language: str
    summary: str
    key_functions: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    complexity_rating: str = Field(default="medium", description="low/medium/high")


class StructuredOutput:
    """Validator for structured (pydantic model-based) output from the agent.

    This ensures the LLM returns strictly-typed function calls and
    multi-agent coordination data, similar to how CrewAI enforces
    output types.
    """

    @staticmethod
    def validate(data: dict[str, Any], model: type[BaseModel]) -> tuple[bool, Any, str]:
        """Validate a dict against a pydantic model.

        Returns:
            (success, validated_model_or_data, error_message)
        """
        try:
            instance = model(**data)
            return True, instance, ""
        except (ValidationError, TypeError) as exc:
            return False, data, str(exc)


# ---------------------------------------------------------------------------
# Repository context provider
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """A file discovered during repo scanning."""

    path: str
    size: int
    lines: int
    language: str
    extension: str
    last_modified: float


class RepoContext:
    """Repository context provider — multi-file codebase awareness.

    Scans the project directory and maintains an index of files, their
    languages, sizes, and modification times.  Provides search and
    retrieval capabilities for the agent to understand the codebase
    before making changes.

    This supports the "Repository Contexting" capability (Layer 4),
    similar to how Aider or Repo-Prompt provide git-aware multi-file
    diffing and autonomous refactoring.
    """

    SUPPORTED_LANGUAGES = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".sh": "bash",
        ".bash": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".cfg": "ini",
        ".md": "markdown",
        ".sql": "sql",
        ".css": "css",
        ".scss": "scss",
        ".html": "html",
    }

    IGNORED_DIRS = {
        "__pycache__", ".git", ".venv", "venv", "env", ".env",
        "node_modules", ".idea", ".vscode", "*.egg-info", ".mypy_cache",
        ".ruff_cache", ".pytest_cache", "legacy", "experiments",
        "datasets/processed", "datasets/raw", "static/dist",
    }

    def __init__(self, repo_root: str | None = None, max_files: int = 500) -> None:
        self._repo_root = repo_root or os.getcwd()
        self._max_files = max_files
        self._index: dict[str, FileEntry] = {}
        self._last_scan = 0.0
        self._scan()

    def _scan(self) -> None:
        """Scan the repository and build the file index."""
        self._index.clear()
        count = 0

        for root, dirs, files in os.walk(self._repo_root):
            # Filter ignored directories
            dirs[:] = [
                d for d in dirs
                if not any(
                    d.lower() == ig.lower() or ig.replace("*", "") in d.lower()
                    for ig in self.IGNORED_DIRS
                )
            ]

            for fname in files:
                if count >= self._max_files:
                    break

                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_LANGUAGES:
                    continue

                try:
                    stat = os.stat(fpath)
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        lines = sum(1 for _ in f)
                    entry = FileEntry(
                        path=os.path.relpath(fpath, self._repo_root),
                        size=stat.st_size,
                        lines=lines,
                        language=self.SUPPORTED_LANGUAGES[ext],
                        extension=ext,
                        last_modified=stat.st_mtime,
                    )
                    self._index[fpath] = entry
                    count += 1
                except (OSError, PermissionError):
                    continue

        self._last_scan = time.time()

    def refresh(self) -> None:
        """Re-scan the repository."""
        self._scan()

    @property
    def repo_root(self) -> str:
        return self._repo_root

    @property
    def file_count(self) -> int:
        return len(self._index)

    def search(self, query: str, limit: int = 20) -> list[FileEntry]:
        """Search for files by name or path containing the query."""
        query_lower = query.lower()
        results = [
            entry for entry in self._index.values()
            if query_lower in entry.path.lower()
        ]
        results.sort(key=lambda e: e.last_modified, reverse=True)
        return results[:limit]

    def get_file(self, path: str) -> FileEntry | None:
        """Get file info by relative path."""
        full_path = os.path.join(self._repo_root, path)
        return self._index.get(full_path)

    def read_file(self, path: str, max_lines: int = 500) -> str:
        """Read a file's contents, limited to max_lines."""
        full_path = os.path.join(self._repo_root, path)
        try:
            with open(full_path, encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"  ... ({max_lines} lines shown, file truncated)")
                        break
                    lines.append(line)
                return "".join(lines)
        except (FileNotFoundError, PermissionError) as exc:
            return f"Error: {exc}"

    def get_language_summary(self) -> dict[str, int]:
        """Return a count of files by language."""
        summary: dict[str, int] = {}
        for entry in self._index.values():
            summary[entry.language] = summary.get(entry.language, 0) + 1
        return summary

    def get_repo_hash(self) -> str:
        """Compute a hash of the current repo state (file paths + sizes)."""
        parts = []
        for entry in sorted(self._index.values(), key=lambda e: e.path):
            parts.append(f"{entry.path}:{entry.size}:{entry.last_modified}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def get_diff_context(
        self,
        file_path: str,
        new_content: str,
    ) -> str:
        """Generate a diff context for a proposed file change.

        Returns a string showing the file's current content alongside
        the proposed changes, useful for the agent to review before
        applying.
        """
        current = self.read_file(file_path)
        return (
            f"=== Current file: {file_path} ===\n{current}\n\n"
            f"=== Proposed new content: ===\n{new_content}\n"
        )


# ---------------------------------------------------------------------------
# Docker sandbox adapter (E2B-compatible)
# ---------------------------------------------------------------------------

class DockerSandbox:
    """Docker-based sandbox for executing untrusted LLM-generated code.

    Provides an E2B-compatible interface.  If Docker is not available,
    falls back to local subprocess execution with the platform's
    SandboxExecutor.

    Usage::

        sandbox = DockerSandbox()
        result = sandbox.run("print('hello')", language='python')
    """

    def __init__(
        self,
        image: str = "python:3.13-slim",
        timeout: float = 30.0,
        memory_limit: str = "512m",
        fallback_to_local: bool = True,
    ) -> None:
        self._image = image
        self._timeout = timeout
        self._memory_limit = memory_limit
        self._fallback_to_local = fallback_to_local
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def is_docker_available(self) -> bool:
        return self._docker_available

    def run(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute code in a Docker container.

        Falls back to local subprocess execution if Docker is unavailable
        and fallback_to_local is True.
        """
        t0 = time.monotonic()

        if language != "python":
            return ExecutionResult(
                tool_name="docker_sandbox",
                success=False,
                error=f"Language '{language}' not supported (python only)",
                elapsed_seconds=time.monotonic() - t0,
            )

        if not self._docker_available:
            if not self._fallback_to_local:
                return ExecutionResult(
                    tool_name="docker_sandbox",
                    success=False,
                    error="Docker not available and fallback disabled",
                    elapsed_seconds=time.monotonic() - t0,
                )
            # Fallback: local subprocess
            return self._run_local(code, timeout or self._timeout)

        # Docker-based execution
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(textwrap.dedent(code))
                script_path = f.name

            # Copy script into container and run
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--memory", self._memory_limit,
                    "--network", "none",
                    "-v", f"{script_path}:/script.py:ro",
                    self._image,
                    "python", "/script.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout or self._timeout,
            )
            return ExecutionResult(
                tool_name="docker_sandbox",
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                elapsed_seconds=time.monotonic() - t0,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                tool_name="docker_sandbox",
                success=False,
                error="Execution timed out",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="docker_sandbox",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
        finally:
            if script_path:
                os.unlink(script_path)

    def _run_local(self, code: str, timeout: float) -> ExecutionResult:
        """Execute Python code locally as a fallback."""
        t0 = time.monotonic()
        script_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(textwrap.dedent(code))
                script_path = f.name

            import sys
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return ExecutionResult(
                tool_name="docker_sandbox",
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
                elapsed_seconds=time.monotonic() - t0,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                tool_name="docker_sandbox",
                success=False,
                error="Execution timed out",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="docker_sandbox",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
        finally:
            if script_path and os.path.exists(script_path):
                os.unlink(script_path)


# ---------------------------------------------------------------------------
# Coder provider (registers all coding capabilities as agent tools)
# ---------------------------------------------------------------------------

class CoderProvider(ToolProvider):
    """Provider for code interpretation and repository tooling.

    Bundles:
        - Local code interpreter (``run_python_code``)
        - Repository context search (``repo_search``)
        - Repository file read (``repo_read``)
        - Structured output validation (``validate_json``)
        - Docker/E2B sandbox (``run_in_sandbox``)
    """

    def __init__(
        self,
        repo_root: str | None = None,
        docker_sandbox: DockerSandbox | None = None,
        sandbox: Any = None,
    ) -> None:
        self._repo = RepoContext(repo_root) if repo_root else None
        self._docker_sandbox = docker_sandbox or DockerSandbox()
        self._sandbox = sandbox

    def get_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="run_python_code",
                description=(
                    "Execute Python code in a sandboxed subprocess. "
                    "Returns stdout output. Timeout: 30s. "
                    "Use for calculations, data processing, prototyping."
                ),
                parameters={"code": "str - Python code to execute"},
                tags=["execution", "python", "code"],
                risk_level="medium",
                permission_required="execute",
            ),
            ToolSpec(
                name="repo_search",
                description=(
                    "Search the codebase for files by name or path. "
                    "Returns matching files with language and size info."
                ),
                parameters={"query": "str - search term (file name or path)"},
                tags=["repo", "search", "context"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="repo_read",
                description=(
                    "Read a file's contents from the repository. "
                    "Returns up to 500 lines of content."
                ),
                parameters={"path": "str - relative path within the repo"},
                tags=["repo", "read", "context"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="repo_summary",
                description=(
                    "Get a summary of the repository: file count, "
                    "language breakdown, and repo hash."
                ),
                parameters={},
                tags=["repo", "summary"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="validate_json",
                description=(
                    "Validate a JSON string against a schema. "
                    "Returns parsed JSON if valid."
                ),
                parameters={
                    "json_str": "str - JSON string to validate",
                    "required_keys": "list - keys that must be present",
                },
                tags=["validation", "json"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="run_in_sandbox",
                description=(
                    "Execute Python code in a Docker container (E2B-compatible). "
                    "Falls back to local execution if Docker is unavailable."
                ),
                parameters={"code": "str - Python code to execute in sandbox"},
                tags=["sandbox", "docker", "security"],
                risk_level="high",
                permission_required="execute",
            ),
        ]

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()

        if name == "run_python_code":
            return self._run_python_code(**kwargs)
        elif name == "repo_search":
            return self._repo_search(**kwargs)
        elif name == "repo_read":
            return self._repo_read(**kwargs)
        elif name == "repo_summary":
            return self._repo_summary(**kwargs)
        elif name == "validate_json":
            return self._validate_json(**kwargs)
        elif name == "run_in_sandbox":
            return self._run_in_sandbox(**kwargs)

        return ExecutionResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: {name}",
            elapsed_seconds=time.monotonic() - t0,
        )

    def _run_python_code(self, code: str, **kwargs: Any) -> ExecutionResult:
        """Execute Python code locally."""
        t0 = time.monotonic()

        code = textwrap.dedent(code).strip()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with tempfile.TemporaryDirectory(prefix="silverwing_coder_") as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, {})  # noqa: S102
            except Exception as exc:
                import traceback
                traceback.print_exc(file=stderr_buf)
                return ExecutionResult(
                    tool_name="run_python_code",
                    success=False,
                    error=str(exc),
                    elapsed_seconds=time.monotonic() - t0,
                    metadata={
                        "stdout": stdout_buf.getvalue(),
                        "stderr": stderr_buf.getvalue()[:2000],
                    },
                )
            finally:
                os.chdir(original_cwd)

        return ExecutionResult(
            tool_name="run_python_code",
            success=True,
            output=stdout_buf.getvalue() or stderr_buf.getvalue(),
            elapsed_seconds=time.monotonic() - t0,
            metadata={
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue()[:2000],
            },
        )

    def _repo_search(self, query: str, limit: int = 20, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if self._repo is None:
            self._repo = RepoContext()
        results = self._repo.search(query, limit=limit)
        if not results:
            return ExecutionResult(
                tool_name="repo_search",
                success=True,
                output="No files found matching query.",
                elapsed_seconds=time.monotonic() - t0,
            )
        lines = []
        for entry in results:
            lines.append(f"  {entry.path} ({entry.language}, {entry.lines} lines, {entry.size} bytes)")
        return ExecutionResult(
            tool_name="repo_search",
            success=True,
            output=f"Found {len(results)} file(s):\n" + "\n".join(lines),
            elapsed_seconds=time.monotonic() - t0,
        )

    def _repo_read(self, path: str, max_lines: int = 500, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if self._repo is None:
            self._repo = RepoContext()
        content = self._repo.read_file(path, max_lines=max_lines)
        return ExecutionResult(
            tool_name="repo_read",
            success=True,
            output=content,
            elapsed_seconds=time.monotonic() - t0,
        )

    def _repo_summary(self, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        if self._repo is None:
            self._repo = RepoContext()
        lang_summary = self._repo.get_language_summary()
        lang_lines = []
        for lang, count in sorted(lang_summary.items(), key=lambda x: -x[1]):
            lang_lines.append(f"  {lang}: {count} files")
        output = (
            f"Repository: {self._repo.repo_root}\n"
            f"Total files: {self._repo.file_count}\n"
            f"Repo hash: {self._repo.get_repo_hash()}\n\n"
            f"Languages:\n" + "\n".join(lang_lines)
        )
        return ExecutionResult(
            tool_name="repo_summary",
            success=True,
            output=output,
            elapsed_seconds=time.monotonic() - t0,
        )

    def _validate_json(self, json_str: str, required_keys: list[str] | None = None, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return ExecutionResult(
                tool_name="validate_json",
                success=False,
                error=f"Invalid JSON: {exc}",
                elapsed_seconds=time.monotonic() - t0,
            )

        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return ExecutionResult(
                    tool_name="validate_json",
                    success=False,
                    error=f"Missing required keys: {missing}",
                    elapsed_seconds=time.monotonic() - t0,
                )

        return ExecutionResult(
            tool_name="validate_json",
            success=True,
            output=f"Valid JSON with {len(data)} top-level keys",
            elapsed_seconds=time.monotonic() - t0,
            metadata={"data": data},
        )

    def _run_in_sandbox(self, code: str, **kwargs: Any) -> ExecutionResult:
        """Execute code in Docker sandbox (E2B-compatible)."""
        result = self._docker_sandbox.run(code)
        result.tool_name = "run_in_sandbox"
        return result
