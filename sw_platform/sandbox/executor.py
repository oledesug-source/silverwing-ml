"""Sandboxed execution with resource limits.

``SandboxExecutor`` wraps every capability execution with a timeout,
resource boundary, and error handler.  The LLM never runs code directly —
every execution passes through here.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from intelligence.tools.protocol import ToolResult


class _TimeoutError(Exception):
    """Raised when a sandboxed call exceeds its timeout."""


def _run_with_timeout(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout: float,
) -> Any:
    """Run *fn(**kwargs)* with a timeout via a daemon thread."""
    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(fn(**kwargs))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise _TimeoutError(f"Execution timed out after {timeout}s")
    if error:
        raise error[0]
    return result[0] if result else None


@dataclass
class ResourceLimits:
    """Resource constraints enforced by the sandbox.

    Attributes:
        max_memory_bytes:   Maximum memory the call may use (advisory).
        max_file_size_bytes: Maximum file size for read/write operations.
        max_execution_time: Wall-clock timeout in seconds.
        allowed_paths:      If non-empty, only these path prefixes are allowed.
        blocked_paths:      These path prefixes are always blocked.
        network_allowed:    Whether network access is permitted.
    """

    max_memory_bytes: int | None = None
    max_file_size_bytes: int | None = None
    max_execution_time: float = 30.0
    allowed_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=list)
    network_allowed: bool = False


class SandboxExecutor:
    """Safe execution environment for capability calls.

    Every capability execution goes through the sandbox, which enforces
    timeouts, path restrictions, and error boundaries.

    Usage::

        sandbox = SandboxExecutor(ResourceLimits(max_execution_time=5.0))
        result = sandbox.execute(
            lambda expression: str(eval(expression)),
            cap_id="calculator",
            expression="2+2",
        )
        # result.output == "4"
    """

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self._limits = limits or ResourceLimits()

    @property
    def limits(self) -> ResourceLimits:
        return self._limits

    def check_path(self, path: str) -> tuple[bool, str]:
        """Validate a file path against allowed/blocked lists.

        Returns ``(ok, reason)``.
        """
        normalized = path.replace("\\", "/").lower()

        for blocked in self._limits.blocked_paths:
            if normalized.startswith(blocked.lower()):
                return False, f"Path blocked: {path}"

        if self._limits.allowed_paths:
            for allowed in self._limits.allowed_paths:
                if normalized.startswith(allowed.lower()):
                    return True, "allowed"
            return False, f"Path not in allowed list: {path}"

        return True, "allowed"

    def check_file_size(self, size_bytes: int) -> tuple[bool, str]:
        """Validate file size against limits."""
        if self._limits.max_file_size_bytes is not None:
            if size_bytes > self._limits.max_file_size_bytes:
                return False, (
                    f"File too large: {size_bytes} bytes "
                    f"(limit: {self._limits.max_file_size_bytes})"
                )
        return True, "allowed"

    def execute(
        self,
        fn: Callable[..., Any],
        cap_id: str = "unknown",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute *fn* inside the sandbox.

        Returns a ``ToolResult`` regardless of success or failure.
        """
        try:
            output = _run_with_timeout(
                fn, kwargs, self._limits.max_execution_time,
            )
            return ToolResult(
                tool_name=cap_id,
                output=str(output) if output is not None else "",
                success=True,
            )
        except _TimeoutError as exc:
            return ToolResult(
                tool_name=cap_id,
                output="",
                success=False,
                error=f"Sandbox timeout: {exc}",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=cap_id,
                output="",
                success=False,
                error=f"Sandbox error: {exc}",
            )
