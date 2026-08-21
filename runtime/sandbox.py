"""Safe execution wrapper for tool calls.

``Sandbox`` wraps every capability execution with a timeout and error
boundary.  On Windows it uses ``threading.Timer``; on POSIX it can also
use ``signal.alarm``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from intelligence.tools.protocol import ToolResult


class _TimeoutError(Exception):
    """Raised when a sandboxed call exceeds its timeout."""


def _run_with_timeout(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout: float,
) -> Any:
    """Run *fn(**kwargs)* with a timeout.

    Uses ``threading.Timer`` (cross-platform) with a sentinel to propagate
    the result back to the caller.
    """
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

    if not result:
        return None
    return result[0]


@dataclass
class Sandbox:
    """Safe execution wrapper for tool calls.

    Usage::

        sandbox = Sandbox(timeout_seconds=5.0)
        result = sandbox.execute(
            lambda expression: str(eval(expression)),
            expression="2+2",
        )
        # result.output == "4"
    """

    timeout_seconds: float = 30.0

    def execute(
        self,
        fn: Callable[..., Any],
        tool_name: str = "unknown",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute *fn* inside the sandbox.

        Returns a ``ToolResult`` regardless of success or failure.
        """
        try:
            output = _run_with_timeout(fn, kwargs, self.timeout_seconds)
            return ToolResult(
                tool_name=tool_name,
                output=str(output) if output is not None else "",
                success=True,
            )
        except _TimeoutError as exc:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=f"Sandbox timeout: {exc}",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=f"Sandbox error: {exc}",
            )
