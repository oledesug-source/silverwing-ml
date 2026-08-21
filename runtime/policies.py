"""Safety and audit policies.

``PolicyEngine`` enforces round limits, produces audit logs, and decides
when the orchestration loop should stop.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .context import RequestContext

logger = logging.getLogger(__name__)

# Patterns that signal the model is done (no more tool calls expected).
_DONE_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Final Answer|Answer|Result)\s*:", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:Therefore|Thus|So|In conclusion)\b", re.IGNORECASE),
]


@dataclass
class PolicyEngine:
    """Safety and audit policies for the orchestration loop.

    Attributes:
        max_rounds: Maximum number of tool-call rounds per request.
        audit_log: Whether to emit structured audit log entries.
    """

    max_rounds: int = 5
    audit_log: bool = True

    def check_round_limit(self, context: RequestContext) -> bool:
        """Return True if the round limit has been reached."""
        rounds = len(context.tool_results)
        return rounds >= self.max_rounds

    def should_stop(self, context: RequestContext, model_output: str) -> tuple[bool, str]:
        """Decide whether the orchestration loop should terminate.

        Returns ``(should_stop, reason)``.
        """
        # Round limit
        if self.check_round_limit(context):
            return True, "round_limit"

        # No tool calls in the output — model is done
        from intelligence.tools.protocol import TOOL_CALL_PATTERN
        if not TOOL_CALL_PATTERN.search(model_output):
            return True, "no_tool_calls"

        return False, "continue"

    def audit(self, context: RequestContext, action: str, detail: str = "") -> None:
        """Emit an audit log entry."""
        if not self.audit_log:
            return
        logger.info(
            "AUDIT request=%s action=%s detail=%s elapsed=%.2f",
            context.request_id,
            action,
            detail,
            time.time() - context.timestamp,
        )
