"""Platform policy engine (Phase 10).

Runtime-configurable policies over named resource categories.  Every
capability action passes through the policy engine *before* the permission
engine and *before* sandbox execution:

    action
      -> policy.evaluate(category)   -> allow | deny | require_approval
      -> permission check            -> may raise (level-based)
      -> approval (if require_approval) -> pending until granted
      -> sandbox execution

Policies are data, not hard-coded model behaviour.  Defaults are safe for a
local self-hosted platform: read/math are allowed automatically; writes,
deletes, credentials and autonomous execution are denied by default and must
be explicitly relaxed by an operator.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sw_platform.audit.events import AuditLog
    from sw_platform.capabilities.schema import CapabilitySchema

logger = logging.getLogger(__name__)

__all__ = [
    "PolicyDecision",
    "PolicyRule",
    "PolicyEngine",
    "RESOURCE_CATEGORIES",
]

RESOURCE_CATEGORIES = [
    "read",
    "write",
    "delete",
    "execute",
    "network",
    "external_communication",
    "credentials",
    "sensitive_resources",
    "autonomous_execution",
]


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyRule:
    """A single policy rule mapping a resource category to a decision.

    Attributes:
        category: Resource category (see :data:`RESOURCE_CATEGORIES`).
        decision: ``allow`` | ``deny`` | ``require_approval``.
        condition: Optional callable ``(capability, context) -> bool``.  When
            provided the rule only applies when the condition is true.
        constraints: Arbitrary metadata (e.g. rate limits, path allowlists).
    """

    category: str
    decision: PolicyDecision = PolicyDecision.DENY
    condition: Callable[..., bool] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

    def applies(self, capability: CapabilitySchema, context: Any = None) -> bool:
        if self.condition is None:
            return True
        try:
            return bool(self.condition(capability, context))
        except Exception:
            return False


def _category_for_capability(capability: CapabilitySchema) -> str:
    """Map a capability to its primary resource category.

    Resolution order:
      1. Explicit ``category`` in capability metadata.
      2. Tag overlap with category names.
      3. Risk/permission heuristics.
      4. Default to ``read``.
    """
    meta = capability.metadata if hasattr(capability, "metadata") else {}
    if isinstance(meta, dict):
        cat = meta.get("category")
        if cat:
            return cat

    tagset = {t.lower() for t in capability.tags}
    for cat in RESOURCE_CATEGORIES:
        if cat in tagset:
            return cat

    if "math" in tagset or "safe" in tagset or "calc" in tagset:
        return "read"
    if "write" in tagset or "file" in tagset:
        return "write"
    if "delete" in tagset:
        return "delete"
    if "network" in tagset or "web" in tagset or "http" in tagset:
        return "network"
    if "credential" in tagset or "secret" in tagset or "auth" in tagset:
        return "credentials"
    if capability.risk_level == "critical":
        return "execute"
    return "read"


class PolicyEngine:
    """Runtime policy engine.

    Usage::

        engine = PolicyEngine()            # safe defaults
        engine.set_rule("write", PolicyDecision.REQUIRE_APPROVAL)
        decision = engine.evaluate(capability)  # -> allow
    """

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._rules: dict[str, PolicyRule] = {}
        self._audit = audit
        if rules:
            for r in rules:
                self._rules[r.category] = r
        else:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Apply safe default policies."""
        self._rules = {
            "read": PolicyRule("read", PolicyDecision.ALLOW),
            "write": PolicyRule("write", PolicyDecision.REQUIRE_APPROVAL),
            "delete": PolicyRule("delete", PolicyDecision.DENY),
            "execute": PolicyRule("execute", PolicyDecision.REQUIRE_APPROVAL),
            "network": PolicyRule("network", PolicyDecision.REQUIRE_APPROVAL),
            "external_communication": PolicyRule(
                "external_communication", PolicyDecision.REQUIRE_APPROVAL
            ),
            "credentials": PolicyRule("credentials", PolicyDecision.DENY),
            "sensitive_resources": PolicyRule(
                "sensitive_resources", PolicyDecision.REQUIRE_APPROVAL
            ),
            "autonomous_execution": PolicyRule(
                "autonomous_execution", PolicyDecision.DENY
            ),
        }

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def set_rule(self, category: str, decision: PolicyDecision, **kw: Any) -> None:
        self._rules[category] = PolicyRule(category, decision, **kw)

    def get_rule(self, category: str) -> PolicyRule | None:
        return self._rules.get(category)

    def list_rules(self) -> list[PolicyRule]:
        return list(self._rules.values())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def category_for(self, capability: CapabilitySchema) -> str:
        return _category_for_capability(capability)

    def evaluate(
        self, capability: CapabilitySchema, context: Any = None
    ) -> tuple[PolicyDecision, str]:
        """Evaluate the policy for *capability*.

        Returns ``(decision, reason)``.
        """
        if not capability.enabled:
            self._record_audit(
                context, capability, "policy", "denied", "capability disabled"
            )
            return PolicyDecision.DENY, "capability_disabled"

        category = self.category_for(capability)
        rule = self._rules.get(category)
        if rule is None:
            self._record_audit(
                context, capability, "policy", "denied", f"no rule for {category}"
            )
            return PolicyDecision.DENY, f"no_policy_for_{category}"

        if not rule.applies(capability, context):
            self._record_audit(
                context, capability, "policy", "allowed", "rule condition not matched"
            )
            return PolicyDecision.ALLOW, "condition_not_matched"

        self._record_audit(
            context, capability, "policy", rule.decision.value, category
        )
        return rule.decision, category

    def _record_audit(
        self,
        context: Any,
        capability: CapabilitySchema,
        action: str,
        status: str,
        detail: str,
    ) -> None:
        if self._audit is None:
            return
        from sw_platform.audit.events import AuditEvent
        from sw_platform.context.models import RequestContext

        request_id = ""
        session_id = ""
        if isinstance(context, RequestContext):
            request_id = context.request_id
            session_id = context.session.session_id
        self._audit.record(AuditEvent(
            request_id=request_id,
            session_id=session_id,
            action=action,
            capability_id=capability.name,
            status=status,
            detail=detail,
        ))


def default_policy_engine(audit: AuditLog | None = None) -> PolicyEngine:
    """Construct a PolicyEngine with safe defaults."""
    return PolicyEngine(audit=audit)
