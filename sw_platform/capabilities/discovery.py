"""Task-aware capability discovery.

``CapabilityDiscovery`` selects the most relevant capabilities for a
given task description using tag matching and metadata scoring — no
AI dependency required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import CapabilityRegistry
from .schema import CapabilitySchema

if TYPE_CHECKING:
    from sw_platform.context.models import RequestContext


class CapabilityDiscovery:
    """Discovers capabilities relevant to a task.

    Usage::

        discovery = CapabilityDiscovery(registry)
        relevant = discovery.find_for_task("calculate 2 + 2")
    """

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def find_for_task(
        self,
        task_description: str,
        context: RequestContext | None = None,
    ) -> list[CapabilitySchema]:
        """Find capabilities relevant to *task_description*.

        Scoring heuristic:
        1. Filter to enabled capabilities only.
        2. Score each capability by keyword overlap with the task.
        3. Boost capabilities whose tags appear in the task.
        4. Return sorted by score descending.
        """
        task_lower = task_description.lower()
        task_words = set(task_lower.split())
        candidates = list(self._registry.list(enabled_only=True))

        scored: list[tuple[float, CapabilitySchema]] = []
        for cap in candidates:
            score = 0.0

            name_words = set(cap.name.lower().replace("_", " ").replace("-", " ").split())
            score += len(task_words & name_words) * 2.0

            desc_words = set(cap.description.lower().split())
            score += len(task_words & desc_words) * 1.0

            for tag in cap.tags:
                if tag.lower() in task_lower:
                    score += 3.0

            if score > 0:
                scored.append((score, cap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cap for _, cap in scored]
