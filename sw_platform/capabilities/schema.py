"""Capability schema definition.

Every capability in the platform is described by a ``CapabilitySchema`` that
carries its identity, versioning, input/output contracts, permission
requirements, risk level, and execution constraints.  The LLM *proposes*
actions against these schemas — the runtime *decides* whether to execute.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CapabilitySchema:
    """Schema describing a registered capability.

    Attributes:
        id:          Unique identifier (auto-generated if omitted).
        name:        Human-readable name (must be unique in the registry).
        version:     Semver string (e.g. ``"1.0.0"``).
        description: Short description of what the capability does.
        input_schema:  JSON-Schema-like dict describing accepted inputs.
        output_schema: JSON-Schema-like dict describing outputs.
        permissions_required: Minimum permission level(s) needed
                              (e.g. ``["L0"]``, ``["L0", "L2"]``).
        risk_level:  One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        timeout_seconds: Maximum wall-clock time for a single execution.
        execution_mode: ``"sync"`` or ``"async"``.
        enabled:     Whether the capability is currently available.
        capability_type: ``"tool"``, ``"reasoning"``, or ``"generation"``.
        fn:          The callable implementation (``None`` for abstract caps).
        tags:        Freeform tags for discovery (e.g. ``["math", "safe"]``).
        source:      Origin: ``"builtin"``, ``"user"``, or ``"external"``.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    permissions_required: list[str] = field(default_factory=lambda: ["L0"])
    risk_level: str = "low"
    timeout_seconds: float = 30.0
    execution_mode: str = "sync"
    enabled: bool = True
    capability_type: str = "tool"
    fn: Callable[..., Any] | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "builtin"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def matches_permission(self, level_str: str) -> bool:
        """Return True if *level_str* satisfies this capability's requirement."""
        level_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
        required_max = max(
            (level_order.get(p, 0) for p in self.permissions_required),
            default=0,
        )
        provided = level_order.get(level_str, 0)
        return provided >= required_max
