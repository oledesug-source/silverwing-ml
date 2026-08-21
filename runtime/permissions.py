"""Per-tool access control.

``PermissionPolicy`` defines which capabilities are allowed or denied.
``PermissionCheck`` evaluates the policy before a capability is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PermissionPolicy:
    """Per-tool permission rules.

    Attributes:
        allowed_tools: If set, *only* these tools may be used.
                       If ``None``, all tools are allowed (subject to denied_tools).
        denied_tools:  Tools that are always denied, even if in allowed_tools.
        require_sandbox: Tools that must execute inside the sandbox.
    """

    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(default_factory=set)
    require_sandbox: set[str] = field(default_factory=set)


class PermissionCheck:
    """Checks permissions before a capability is executed.

    Usage::

        policy = PermissionPolicy(denied_tools={"shell", "network"})
        check = PermissionCheck(policy)
        ok, reason = check.is_allowed("calculator")
        assert ok
        ok, reason = check.is_allowed("shell")
        assert not ok
    """

    def __init__(self, policy: PermissionPolicy) -> None:
        self._policy = policy

    def is_allowed(self, capability_name: str) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for a capability."""
        if capability_name in self._policy.denied_tools:
            return False, f"Tool '{capability_name}' is denied by policy"

        if self._policy.allowed_tools is not None:
            if capability_name not in self._policy.allowed_tools:
                return False, (
                    f"Tool '{capability_name}' is not in the allowed list"
                )

        return True, "allowed"

    def needs_sandbox(self, capability_name: str) -> bool:
        """Return True if the capability must run inside the sandbox."""
        return capability_name in self._policy.require_sandbox
