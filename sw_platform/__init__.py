"""SilverWing Runtime v1 — Controlled Intelligence Platform.

The LLM proposes — the platform decides and executes.

Every capability is schema-defined, permission-gated, resource-bounded,
and audit-logged.

Layer 4 integration: the platform's policy engine, approval manager, and
model/database abstractions (from ``silverwing_platform``) are wired into
the orchestrator so that every capability action passes through
policy → permission → approval → sandbox → audit before execution.
"""

from sw_platform.audit.events import AuditEvent, AuditLog
from sw_platform.capabilities.registry import CapabilityRegistry
from sw_platform.capabilities.schema import CapabilitySchema
from sw_platform.context.builder import ContextBuilder
from sw_platform.context.models import RequestContext, SessionState
from sw_platform.orchestration.orchestrator import ChatRequest, ChatResponse, Orchestrator
from sw_platform.permissions.policy import (
    PermissionEvaluator,
    PermissionLevel,
    PermissionPolicy,
)
from sw_platform.sandbox.executor import ResourceLimits, SandboxExecutor

from silverwing_platform.approvals import ApprovalManager
from silverwing_platform.policies import PolicyDecision, PolicyEngine

from sw_platform.tools.mlops import register_mlops_capabilities

__all__ = [
    "AuditEvent",
    "AuditLog",
    "ApprovalManager",
    "CapabilityRegistry",
    "CapabilitySchema",
    "ChatRequest",
    "ChatResponse",
    "ContextBuilder",
    "Orchestrator",
    "PermissionEvaluator",
    "PermissionLevel",
    "PermissionPolicy",
    "PolicyDecision",
    "PolicyEngine",
    "RequestContext",
    "ResourceLimits",
    "SandboxExecutor",
    "SessionState",
    "register_mlops_capabilities",
]
