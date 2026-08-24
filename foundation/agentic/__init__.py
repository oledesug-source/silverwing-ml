"""Agentic AI stack for Silverwing — six capability levels.

L1 basic_responder   single-shot generation
L2 router            intent classification dispatching to handlers
L3 tool_calling      ReAct-style loop over platform tools
L4 multi_agent       role-specialised agents coordinated by an orchestrator
L5 autonomous        goal decomposition into a guarded execution plan
L6 loop_engineering  OODA outer loop with reflection, memory and metrics
"""

from __future__ import annotations

from .backend import DeterministicBackend, HttpOpenAICompat, LlmBackend, ScriptedBackend
from .engine import AgenticEngine
from .levels import AgentLevel, AgentTrace, TraceStep

__all__ = [
    "AgentLevel",
    "AgentTrace",
    "AgenticEngine",
    "DeterministicBackend",
    "HttpOpenAICompat",
    "LlmBackend",
    "ScriptedBackend",
    "TraceStep",
]
