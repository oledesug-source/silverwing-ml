"""Filesystem access tools — direct access to local file paths (os, shutil).

Provides read/write/list/move/delete operations with path validation
against allowed prefixes and optional sandbox enforcement.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from sw_platform.harness.core import ExecutionResult, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)


class FilesystemProvider(ToolProvider):
    """Provider for filesystem operation tools.

    Parameters:
        allowed_paths: Path prefixes that are permitted for access.
            Empty list means no restriction (sandbox must handle this).
        sandbox: Optional SandboxExecutor for path validation.
        max_file_size: Maximum file size for writes in bytes.
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        sandbox: Any = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    ) -> None:
        self._allowed_paths = allowed_paths or []
        self._sandbox = sandbox
        self._max_file_size = max_file_size

    def get_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="read_file",
                description="Read a text file and return its contents.",
                parameters={"path": "str — absolute or relative file path"},
                tags=["filesystem", "read"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="write_file",
                description="Write content to a file. Creates parent dirs if needed.",
                parameters={
                    "path": "str — file path to write",
                    "content": "str — file content",
                },
                tags=["filesystem", "write"],
                risk_level="medium",
                permission_required="write",
            ),
            ToolSpec(
                name="list_directory",
                description="List contents of a directory.",
                parameters={"path": "str — directory path"},
                tags=["filesystem", "read"],
                risk_level="low",
                permission_required="read",
            ),
            ToolSpec(
                name="move_file",
                description="Move or rename a file.",
                parameters={
                    "src": "str — source path",
                    "dst": "str — destination path",
                },
                tags=["filesystem", "write"],
                risk_level="medium",
                permission_required="write",
            ),
            ToolSpec(
                name="delete_file",
                description="Delete a file.",
                parameters={"path": "str — file path to delete"},
                tags=["filesystem", "write"],
                risk_level="high",
                permission_required="write",
            ),
        ]

    def execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()

        # Delegate to sandbox if available
        if self._sandbox is not None:
            try:
                from sw_platform.sandbox.executor import SandboxExecutor
                if isinstance(self._sandbox, SandboxExecutor):
                    result = self._sandbox.execute(
                        self._do_execute,
                        cap_id=name,
                        name=name,
                        **kwargs,
                    )
                    if result.success:
                        return ExecutionResult(
                            tool_name=name,
                            success=True,
                            output=result.output,
                            elapsed_seconds=time.monotonic() - t0,
                        )
                    else:
                        return ExecutionResult(
                            tool_name=name,
                            success=False,
                            error=result.error,
                            elapsed_seconds=time.monotonic() - t0,
                        )
            except Exception:
                pass  # Fall through to direct execution

        return self._do_execute(name=name, **kwargs)

    def _do_execute(self, name: str, **kwargs: Any) -> ExecutionResult:
        """Direct execution of filesystem operations."""
        t0 = time.monotonic()

        def elapsed() -> float:
            return time.monotonic() - t0

        if name == "read_file":
            return self._read_file(**kwargs)
        elif name == "write_file":
            return self._write_file(**kwargs)
        elif name == "list_directory":
            return self._list_directory(**kwargs)
        elif name == "move_file":
            return self._move_file(**kwargs)
        elif name == "delete_file":
            return self._delete_file(**kwargs)

        return ExecutionResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: {name}",
            elapsed_seconds=elapsed(),
        )

    def _check_path(self, path: str) -> tuple[bool, str, str]:
        """Validate path against allowed_paths.

        Returns (ok, reason, normalized_path).
        """
        resolved = os.path.abspath(path)
        if self._allowed_paths:
            for allowed in self._allowed_paths:
                allowed_resolved = os.path.abspath(allowed)
                if resolved.startswith(allowed_resolved):
                    return True, "allowed", resolved
            return False, f"Path not in allowed list: {path}", resolved
        return True, "allowed", resolved

    def _read_file(self, path: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        ok, reason, resolved = self._check_path(path)
        if not ok:
            return ExecutionResult(
                tool_name="read_file",
                success=False,
                error=reason,
                elapsed_seconds=time.monotonic() - t0,
            )
        try:
            with open(resolved, encoding="utf-8") as f:
                content = f.read()
            return ExecutionResult(
                tool_name="read_file",
                success=True,
                output=content,
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="read_file",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )

    def _write_file(self, path: str, content: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        ok, reason, resolved = self._check_path(path)
        if not ok:
            return ExecutionResult(
                tool_name="write_file",
                success=False,
                error=reason,
                elapsed_seconds=time.monotonic() - t0,
            )
        if len(content) > self._max_file_size:
            return ExecutionResult(
                tool_name="write_file",
                success=False,
                error=f"Content too large: {len(content)} bytes (limit: {self._max_file_size})",
                elapsed_seconds=time.monotonic() - t0,
            )
        try:
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            return ExecutionResult(
                tool_name="write_file",
                success=True,
                output=f"Written {len(content)} bytes to {resolved}",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="write_file",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )

    def _list_directory(self, path: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        ok, reason, resolved = self._check_path(path)
        if not ok:
            return ExecutionResult(
                tool_name="list_directory",
                success=False,
                error=reason,
                elapsed_seconds=time.monotonic() - t0,
            )
        try:
            entries = sorted(os.listdir(resolved))
            dirs = [e for e in entries if os.path.isdir(os.path.join(resolved, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(resolved, e))]
            output = "Directories:\n" + "\n".join(f"  [d] {d}" for d in dirs)
            if files:
                output += "\nFiles:\n" + "\n".join(f"  [f] {f}" for f in files)
            return ExecutionResult(
                tool_name="list_directory",
                success=True,
                output=output,
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="list_directory",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )

    def _move_file(self, src: str, dst: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        ok_src, reason_src, resolved_src = self._check_path(src)
        ok_dst, reason_dst, resolved_dst = self._check_path(dst)
        if not ok_src:
            return ExecutionResult(
                tool_name="move_file",
                success=False,
                error=reason_src,
                elapsed_seconds=time.monotonic() - t0,
            )
        if not ok_dst:
            return ExecutionResult(
                tool_name="move_file",
                success=False,
                error=reason_dst,
                elapsed_seconds=time.monotonic() - t0,
            )
        try:
            Path(resolved_dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(resolved_src, resolved_dst)
            return ExecutionResult(
                tool_name="move_file",
                success=True,
                output=f"Moved {resolved_src} → {resolved_dst}",
                elapsed_seconds=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExecutionResult(
                tool_name="move_file",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )

    def _delete_file(self, path: str, **kwargs: Any) -> ExecutionResult:
        t0 = time.monotonic()
        ok, reason, resolved = self._check_path(path)
        if not ok:
            return ExecutionResult(
                tool_name="delete_file",
                success=False,
                error=reason,
                elapsed_seconds=time.monotonic() - t0,
            )
        try:
            if os.path.isfile(resolved):
                os.remove(resolved)
                return ExecutionResult(
                    tool_name="delete_file",
                    success=True,
                    output=f"Deleted {resolved}",
                    elapsed_seconds=time.monotonic() - t0,
                )
            elif os.path.isdir(resolved):
                shutil.rmtree(resolved)
                return ExecutionResult(
                    tool_name="delete_file",
                    success=True,
                    output=f"Deleted directory {resolved}",
                    elapsed_seconds=time.monotonic() - t0,
                )
            else:
                return ExecutionResult(
                    tool_name="delete_file",
                    success=False,
                    error=f"Path does not exist: {resolved}",
                    elapsed_seconds=time.monotonic() - t0,
                )
        except Exception as exc:
            return ExecutionResult(
                tool_name="delete_file",
                success=False,
                error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
