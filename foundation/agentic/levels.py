"""Level definitions and trace models shared by all agentic levels."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


class AgentLevel(IntEnum):
    """The six agentic capability levels."""

    BASIC_RESPONDER = 1
    ROUTER = 2
    TOOL_CALLING = 3
    MULTI_AGENT = 4
    AUTONOMOUS = 5
    LOOP_ENGINEERING = 6

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS: dict[int, str] = {
    AgentLevel.BASIC_RESPONDER: "L1 basic responder",
    AgentLevel.ROUTER: "L2 router pattern",
    AgentLevel.TOOL_CALLING: "L3 tool calling",
    AgentLevel.MULTI_AGENT: "L4 multi-agent",
    AgentLevel.AUTONOMOUS: "L5 autonomous",
    AgentLevel.LOOP_ENGINEERING: "L6 loop engineering",
}


def coerce_level(value: int | str | AgentLevel) -> AgentLevel:
    if isinstance(value, AgentLevel):
        return value
    if isinstance(value, int):
        return AgentLevel(value)
    text = str(value).strip().lower()
    for level in AgentLevel:
        if text in (str(level.value), level.name.lower(), level.label.split(" ", 1)[1]):
            return level
    raise ValueError(f"unknown agent level: {value!r}")


@dataclass
class TraceStep:
    """One observable step inside an agent run."""

    kind: str
    detail: str = ""
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentTrace:
    """Full observable result of an agentic run."""

    level: AgentLevel
    final_text: str = ""
    success: bool = True
    steps: list[TraceStep] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    session_id: str = ""

    def add(self, kind: str, detail: str = "", **data: object) -> TraceStep:
        step = TraceStep(kind=kind, detail=detail, data=dict(data))
        self.steps.append(step)
        return step

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "level_label": self.level.label,
            "final_text": self.final_text,
            "success": self.success,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "session_id": self.session_id,
            "steps": [s.to_dict() for s in self.steps],
        }
