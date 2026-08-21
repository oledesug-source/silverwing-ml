"""Git command wrapper tools — grant the agent direct access to git commands.

This supports the "Repository Contexting" capability (Layer 4), wrapping
git clone, git commit, git status, git diff, git log, and git blame
so the agent can perform version-control operations autonomously,
similar to how Aider or Repo-Prompt provide git-aware multi-file
diffing and autonomous refactoring.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any

from sw_platform.harness.core import ExecutionResult, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)


class GitProvider(ToolProvider):
    """Provider for git command tools.

    Parameters:
        repo_path: Root directory of the git repository.
            Defaults to the current working directory.
    """

    def __init__(self, repo_path: str | None = None) -> None:
        self._repo_path = repo_path or os.getcwd()

    def get_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="git_status",
                description="Show working tree status - modified, staged, untracked files.",
                parameters={},
                tags=["git", "version-control"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="git_diff",
                description="Show changes between commits/working tree.",
                parameters={
                    "target": "str - 'staged', 'unstaged', or empty for comparison with HEAD",
                },
                tags=["git", "version-control", "diff"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="git_log",
                description="Show commit history.",
                parameters={
                    "flags": "str - additional git log flags (e.g. '--oneline -n 5')",
                },
                tags=["git", "version-control", "history"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="git_add",
                description="Stage files for commit.",
                parameters={
                    "files": "str - space-separated file paths or 'all'",
                },
                tags=["git", "version-control", "staging"],
                risk_level="low",
                permission_required="write",
            ),
            ToolSpec(
                name="git_commit",
                description="Commit staged changes with a message.",
                parameters={"message": "str - commit message"},
                tags=["git", "version-control", "commit"],
                risk_level="medium",
                permission_required="write",
            ),
            ToolSpec(
                name="git_blame",
                description="Show line-by-line blame for a file.",
                parameters={"path": "str - file path to blame"},
                tags=["git", "version-control", "blame"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="git_clone",
                description="Clone a git repository.",
                parameters={
                    "url": "str - repository URL to clone",
                    "dest": "str - destination directory (optional)",
                },
                tags=["git", "version-control", "clone"],
                risk_level="high",
                permission_required="execute",
            ),
        ]

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()

        if name == "git_status":
            result = self._run_git(["status", "-sb"])
            result.tool_name = "git_status"
            return result
        elif name == "git_diff":
            return self._git_diff(**kwargs)
        elif name == "git_log":
            return self._git_log(**kwargs)
        elif name == "git_add":
            return self._git_add(**kwargs)
        elif name == "git_commit":
            return self._git_commit(**kwargs)
        elif name == "git_blame":
            return self._git_blame(**kwargs)
        elif name == "git_clone":
            return self._git_clone(**kwargs)

        return ExecutionResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: {name}",
            elapsed_seconds=time.monotonic() - t0,
        )

    def _run_git(self, args: list[str]) -> ExecutionResult:
        """Run a git command and return the result."""
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self._repo_path,
            )
            return ExecutionResult(
                tool_name=args[0] if args else "git",
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
                elapsed_seconds=time.monotonic() - t0,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                tool_name="git",
                success=False,
                error="Git command timed out after 30s",
                elapsed_seconds=time.monotonic() - t0,
            )
        except FileNotFoundError:
            return ExecutionResult(
                tool_name="git",
                success=False,
                error="git is not installed",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="git",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )

    def _git_diff(self, target: str = "unstaged", **kwargs: Any) -> ExecutionResult:
        if target == "staged":
            args = ["diff", "--cached"]
        elif target == "unstaged":
            args = ["diff"]
        else:
            args = ["diff", "HEAD"]
        result = self._run_git(args)
        result.tool_name = "git_diff"
        return result

    def _git_log(self, flags: str = "--oneline -n 10", **kwargs: Any) -> ExecutionResult:
        flag_list = flags.split()
        result = self._run_git(["log", *flag_list])
        result.tool_name = "git_log"
        return result

    def _git_add(self, files: str = "all", **kwargs: Any) -> ExecutionResult:
        if files.strip() == "all":
            args = ["add", "-A"]
        else:
            args = ["add", *files.split()]
        result = self._run_git(args)
        result.tool_name = "git_add"
        return result

    def _git_commit(self, message: str, **kwargs: Any) -> ExecutionResult:
        result = self._run_git(["commit", "-m", message])
        result.tool_name = "git_commit"
        return result

    def _git_blame(self, path: str, **kwargs: Any) -> ExecutionResult:
        result = self._run_git(["blame", path])
        result.tool_name = "git_blame"
        return result

    def _git_clone(self, url: str, dest: str | None = None, **kwargs: Any) -> ExecutionResult:
        args = ["clone", url]
        if dest:
            args.append(dest)
        result = self._run_git(args)
        result.tool_name = "git_clone"
        return result
