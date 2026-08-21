"""Built-in safe demonstration tools.

Tools in this module are registered by ``register_builtin_tools`` and are
available by default in the Intelligence Runtime.  All tools are designed
to be safe — no ``eval()``, no network access, no file writes.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

from .capabilities import Capability, CapabilityRegistry

# ---------------------------------------------------------------------------
# Safe math evaluator (AST-based, no eval())
# ---------------------------------------------------------------------------

_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node — only numeric operations allowed."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        result = _BINOPS[type(node.op)](left, right)
        if isinstance(result, float) and result == int(result) and not (result == float("inf") or result == float("-inf")):
            return float(int(result))
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a math expression safely using AST parsing.

    Supported: +, -, *, /, //, %, **, parentheses, negative numbers.

    Examples::

        calculator("2 + 3")       # "5"
        calculator("2**10")       # "1024"
        calculator("(3+4) * 2")   # "14"
    """
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty expression")
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    if isinstance(result, float) and result == int(result):
        return str(int(result))
    return str(result)


# ---------------------------------------------------------------------------
# Safe local file reader (read-only, size-limited)
# ---------------------------------------------------------------------------

def read_file(path: str, max_bytes: int = 65536) -> str:
    """Read a local file — read-only, size-limited.

    Args:
        path: File path to read.
        max_bytes: Maximum bytes to read (default 64 KB).

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        ValueError: If the file exceeds max_bytes.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"File too large: {size} bytes (limit: {max_bytes})"
        )
    return p.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_tools(registry: CapabilityRegistry) -> None:
    """Register all built-in safe tools into *registry*."""
    registry.register(Capability(
        name="calculator",
        description="Evaluate a math expression (+, -, *, /, //, %, **, parentheses)",
        parameters={"expression": "math expression string"},
        fn=calculator,
        source="builtin",
        tags=["math", "safe"],
        requires_permission=False,
        timeout_seconds=5.0,
    ))
    registry.register(Capability(
        name="read_file",
        description="Read a local file (read-only, max 64KB)",
        parameters={"path": "file path", "max_bytes": "max bytes to read (optional)"},
        fn=read_file,
        source="builtin",
        tags=["file", "safe"],
        requires_permission=False,
        timeout_seconds=5.0,
    ))
