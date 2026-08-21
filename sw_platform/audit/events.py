"""Structured audit events.

Every action in the platform — capability calls, permission checks,
orchestration decisions — is recorded as an ``AuditEvent``.  The
``AuditLog`` stores events in an in-memory ring buffer and supports
querying by request, capability, or status.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AuditEvent:
    """A single audit event.

    Attributes:
        event_id:      Unique identifier.
        timestamp:     Unix timestamp.
        request_id:    Associated request.
        session_id:    Associated session.
        action:        What happened (e.g. ``"tool_call"``, ``"permission_denied"``).
        capability_id: Which capability was involved.
        status:        ``"pending"``, ``"success"``, ``"denied"``, ``"error"``, ``"timeout"``.
        detail:        Human-readable detail.
        elapsed_ms:    Execution time in milliseconds.
        metadata:      Arbitrary extra data.
    """

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""
    session_id: str = ""
    action: str = ""
    capability_id: str = ""
    status: str = "pending"
    detail: str = ""
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """In-memory ring-buffer audit log.

    Usage::

        log = AuditLog(max_entries=1000)
        log.record(AuditEvent(action="tool_call", capability_id="calculator"))
        recent = log.recent(10)
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max_entries
        self._entries: list[AuditEvent] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def record(self, event: AuditEvent) -> None:
        """Record an audit event, evicting the oldest if at capacity."""
        self._entries.append(event)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def query(
        self,
        request_id: str | None = None,
        capability_id: str | None = None,
        status: str | None = None,
    ) -> list[AuditEvent]:
        """Query events by optional filters (AND logic)."""
        results = self._entries
        if request_id is not None:
            results = [e for e in results if e.request_id == request_id]
        if capability_id is not None:
            results = [e for e in results if e.capability_id == capability_id]
        if status is not None:
            results = [e for e in results if e.status == status]
        return list(results)

    def recent(self, n: int = 50) -> list[AuditEvent]:
        """Return the *n* most recent events."""
        return list(self._entries[-n:])

    def clear(self) -> None:
        """Remove all events."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
