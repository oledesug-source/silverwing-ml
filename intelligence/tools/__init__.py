"""Tool-use protocol (M15.6).

Framework for the model to request and receive results from external tools:
calculator, code execution, web search, file operations, etc.
"""

from .protocol import (
    Tool,
    ToolCall,
    ToolResult,
    ToolRegistry,
    ToolExecutor,
)

__all__ = ["Tool", "ToolCall", "ToolExecutor", "ToolRegistry", "ToolResult"]
