"""Per-request state container.

``RequestContext`` holds all mutable state for a single user interaction:
the message, working memory, accumulated tool results, and audit trail.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from intelligence.memory.context import MemoryEntry, WorkingMemory
from intelligence.tools.protocol import ToolResult


@dataclass
class RequestContext:
    """Per-request state container.

    Created by the ``Orchestrator`` for every incoming user message and
    threaded through the entire orchestration loop.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_message: str = ""
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    capabilities_used: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    max_tool_rounds: int = 5
    metadata: dict = field(default_factory=dict)

    # -- helpers --

    def add_user_message(self) -> None:
        """Store the user message in working memory."""
        self.working_memory.add(MemoryEntry(
            key=f"user-{self.request_id}",
            content=f"User: {self.user_message}",
            importance=1.0,
        ))

    def add_tool_result(self, result: ToolResult) -> None:
        """Accumulate a tool result and store it in working memory."""
        self.tool_results.append(result)
        self.capabilities_used.append(result.tool_name)
        label = "output" if result.success else "error"
        detail = result.output if result.success else result.error
        self.working_memory.add(MemoryEntry(
            key=f"tool-{result.tool_name}-{self.request_id}",
            content=f"Tool {result.tool_name} {label}: {detail}",
            importance=0.9,
        ))

    def add_assistant_message(self, text: str) -> None:
        """Store the assistant response in working memory."""
        self.working_memory.add(MemoryEntry(
            key=f"assistant-{self.request_id}",
            content=f"Assistant: {text}",
            importance=0.8,
        ))
