"""SilverWing Intelligence Runtime v1.

Integrates the existing foundation and intelligence components into a stable
orchestration layer.

Components:
- **capabilities** — capability registry and discovery (promoted from intelligence.tools.protocol)
- **context** — per-request state container
- **permissions** — per-tool access control
- **policies** — rate limiting, audit logging, safety checks
- **sandbox** — safe execution environment for tool calls
- **tools** — built-in safe demonstration tools (calculator, read_file)
- **agents** — agent wrapper combining intelligence modules
- **orchestration** — the core agent loop
- **workflows** — simple sequential workflow engine
- **api** — extended HTTP API for chat and tool execution
"""

from .api import IntelligenceHandler
from .capabilities import Capability, CapabilityRegistry
from .context import RequestContext
from .orchestration import ChatRequest, ChatResponse, Orchestrator
from .permissions import PermissionCheck, PermissionPolicy
from .policies import PolicyEngine
from .sandbox import Sandbox

__all__ = [
    "Capability",
    "CapabilityRegistry",
    "ChatRequest",
    "ChatResponse",
    "Orchestrator",
    "PermissionCheck",
    "PermissionPolicy",
    "PolicyEngine",
    "RequestContext",
    "Sandbox",
]
