"""PydanticAI-based agent harness for the Silverwing platform.

Public API for the harness module.
"""

from sw_platform.harness.agent import (
    AgentResponse,
    HarnessConfig,
    PydanticAgentHarness,
    ToolCallRecord,
    create_harness_agent,
)
from sw_platform.harness.core import (
    ExecutionResult,
    HarnessAgent,
    HarnessContext,
    ToolProvider,
    ToolSpec,
)

__all__ = [
    "AgentResponse",
    "ExecutionResult",
    "HarnessAgent",
    "HarnessConfig",
    "HarnessContext",
    "PydanticAgentHarness",
    "ToolCallRecord",
    "ToolProvider",
    "ToolSpec",
    "create_harness_agent",
]
