"""Context builder — factory methods for RequestContext and system prompts.

``ContextBuilder`` provides convenient constructors that wire up sessions,
request contexts, and system prompts from the current capability registry
and permission level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.memory.context import WorkingMemory

from .models import RequestContext, SessionState

if TYPE_CHECKING:
    from sw_platform.capabilities.registry import CapabilityRegistry
    from sw_platform.permissions.policy import PermissionLevel


class ContextBuilder:
    """Factory for RequestContext and system prompts."""

    @staticmethod
    def from_request(
        message: str,
        max_rounds: int = 5,
        session_id: str = "",
        user_id: str = "",
        metadata: dict | None = None,
    ) -> RequestContext:
        """Build a ``RequestContext`` from a raw user message."""
        session = SessionState(
            session_id=session_id or SessionState().session_id,
            user_id=user_id,
            working_memory=WorkingMemory(max_tokens=512),
        )
        return RequestContext(
            session=session,
            user_message=message,
            max_rounds=max_rounds,
            metadata=metadata or {},
        )

    @staticmethod
    def build_system_prompt(
        registry: CapabilityRegistry,
        permission_level: PermissionLevel | str = "L0",
    ) -> str:
        """Build a system prompt section describing available capabilities.

        Only capabilities matching the permission level are included.
        """
        from sw_platform.permissions.policy import PermissionLevel as PL

        if isinstance(permission_level, str):
            try:
                level = PL(permission_level)
            except ValueError:
                level = PL[permission_level]
        else:
            level = permission_level

        lines = [f"Permission level: {level.value} ({level.name})"]
        caps = registry.list(enabled_only=True)
        if not caps:
            lines.append("No capabilities available.")
        else:
            lines.append("Available capabilities:")
            for cap in caps:
                lines.append(f"- {cap.name} (v{cap.version}): {cap.description}")
                if cap.tags:
                    lines.append(f"  Tags: {', '.join(cap.tags)}")
                lines.append(f"  Risk: {cap.risk_level}, Timeout: {cap.timeout_seconds}s")
            lines.append(
                "\nTo use a capability, output: <tool:name>param=value, ...</tool>"
            )
        return "\n".join(lines)
