"""Context models for session and request state.

``SessionState`` holds persistent state across requests in a session.
``RequestContext`` holds mutable state for a single user interaction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from intelligence.memory.context import MemoryEntry, WorkingMemory
from intelligence.tools.protocol import ToolResult


@dataclass
class SessionState:
    """Persistent state across requests within a session.

    Attributes:
        session_id:     Unique session identifier.
        user_id:        Optional user identifier.
        working_memory: Shared working memory.
        metadata:       Arbitrary session-level metadata.
        created_at:     Unix timestamp of session creation.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: str = ""
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RequestContext:
    """Per-request state container.

    Created by the ``Orchestrator`` for every incoming user message and
    threaded through the entire orchestration loop.

    Attributes:
        request_id:         Unique request identifier.
        session:            The parent session state.
        user_message:       The user's input.
        capabilities_used:  Names of capabilities invoked this request.
        tool_results:       All tool results accumulated this request.
        max_rounds:         Maximum tool-call rounds allowed.
        metadata:           Arbitrary request-level metadata.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session: SessionState = field(default_factory=SessionState)
    user_message: str = ""
    capabilities_used: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    max_rounds: int = 5
    metadata: dict = field(default_factory=dict)

    @property
    def timestamp(self) -> float:
        return self.session.created_at

    @property
    def working_memory(self) -> WorkingMemory:
        return self.session.working_memory

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
