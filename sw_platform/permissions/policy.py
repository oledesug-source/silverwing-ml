"""Permission levels and policy enforcement.

The platform defines six permission levels (L0–L5).  Every capability
declares the minimum level it requires; the policy declares the maximum
level the current session grants.  The evaluator decides at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sw_platform.capabilities.schema import CapabilitySchema


class PermissionLevel(Enum):
    """Permission levels from read-only (L0) to full system (L5)."""

    L0 = "read"        # read-only (calculator, read_file)
    L1 = "write"       # file writes
    L2 = "execute"     # code execution
    L3 = "network"     # network access
    L4 = "admin"       # system administration
    L5 = "system"      # full system access

    @property
    def numeric(self) -> int:
        return int(self.name[1])

    def __ge__(self, other: PermissionLevel) -> bool:
        return self.numeric >= other.numeric

    def __gt__(self, other: PermissionLevel) -> bool:
        return self.numeric > other.numeric

    def __le__(self, other: PermissionLevel) -> bool:
        return self.numeric <= other.numeric

    def __lt__(self, other: PermissionLevel) -> bool:
        return self.numeric < other.numeric


@dataclass
class PermissionPolicy:
    """Defines the permission boundaries for a session or request.

    Attributes:
        level:          Maximum permission level granted.
        allowed_tools:  If set, *only* these tools may be used.
        denied_tools:   Tools that are always denied.
        require_sandbox: Tools that must execute inside the sandbox.
    """

    level: PermissionLevel = PermissionLevel.L0
    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(default_factory=set)
    require_sandbox: set[str] = field(default_factory=set)


class PermissionEvaluator:
    """Evaluates whether a capability is permitted under a policy.

    Usage::

        policy = PermissionPolicy(level=PermissionLevel.L2)
        evaluator = PermissionEvaluator(policy)
        allowed, reason = evaluator.is_allowed(some_capability)
    """

    def __init__(self, policy: PermissionPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    def is_allowed(self, cap: CapabilitySchema) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for a capability."""
        if not cap.enabled:
            return False, f"Capability '{cap.name}' is disabled"

        if cap.name in self._policy.denied_tools:
            return False, f"Tool '{cap.name}' is denied by policy"

        if self._policy.allowed_tools is not None:
            if cap.name not in self._policy.allowed_tools:
                return False, f"Tool '{cap.name}' is not in the allowed list"

        required_level = self._max_required_level(cap)
        if not (self._policy.level >= required_level):
            return False, (
                f"Insufficient permissions: need {required_level.value}, "
                f"have {self._policy.level.value}"
            )

        return True, "allowed"

    def needs_sandbox(self, cap: CapabilitySchema) -> bool:
        """Return True if the capability must run inside the sandbox."""
        return (
            cap.name in self._policy.require_sandbox
            or cap.risk_level in ("high", "critical")
        )

    def get_max_permission_level(self) -> PermissionLevel:
        return self._policy.level

    @staticmethod
    def _max_required_level(cap: CapabilitySchema) -> PermissionLevel:
        level_map = {
            "L0": PermissionLevel.L0,
            "L1": PermissionLevel.L1,
            "L2": PermissionLevel.L2,
            "L3": PermissionLevel.L3,
            "L4": PermissionLevel.L4,
            "L5": PermissionLevel.L5,
        }
        levels = [level_map.get(p, PermissionLevel.L0) for p in cap.permissions_required]
        return max(levels) if levels else PermissionLevel.L0
