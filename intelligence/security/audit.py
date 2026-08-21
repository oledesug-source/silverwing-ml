"""Audit logging for security-critical operations.

Records all authentication, authorization, data access, and
configuration change events with tamper-evident chaining.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuditAction(Enum):
    """Types of auditable actions."""

    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    TOKEN_CREATE = "token.create"
    TOKEN_REVOKE = "token.revoke"
    PERMISSION_DENIED = "permission.denied"
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    CONFIG_CHANGE = "config.change"
    MODEL_ACCESS = "model.access"
    API_CALL = "api.call"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    action: AuditAction
    subject: str = ""
    resource: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    source_ip: str = ""
    entry_id: int = 0
    prev_hash: str = ""

    @property
    def hash(self) -> str:
        """Compute hash of this entry (depends on prev_hash for chaining)."""
        data = json.dumps({
            "action": self.action.value,
            "subject": self.subject,
            "resource": self.resource,
            "timestamp": self.timestamp,
            "success": self.success,
            "prev_hash": self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class AuditLog:
    """Tamper-evident audit log with hash chaining.

    Usage::

        log = AuditLog(max_entries=10000)
        log.record(AuditAction.AUTH_SUCCESS, subject="user-1", resource="/api")
        log.record(AuditAction.DATA_READ, subject="user-1", resource="corpus-v1")

        entries = log.query(action=AuditAction.AUTH_SUCCESS)
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max_entries
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._counter = 0
        self._last_hash = ""
        self._lock = threading.Lock()

    def record(
        self,
        action: AuditAction,
        subject: str = "",
        resource: str = "",
        details: dict[str, Any] | None = None,
        success: bool = True,
        source_ip: str = "",
    ) -> AuditEntry:
        """Record an audit event."""
        with self._lock:
            self._counter += 1
            entry = AuditEntry(
                action=action,
                subject=subject,
                resource=resource,
                details=details or {},
                success=success,
                source_ip=source_ip,
                entry_id=self._counter,
                prev_hash=self._last_hash,
            )
            self._last_hash = entry.hash
            self._entries.append(entry)
            return entry

    def query(
        self,
        action: AuditAction | None = None,
        subject: str | None = None,
        resource: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters."""
        results: list[AuditEntry] = []
        with self._lock:
            for entry in reversed(self._entries):
                if action and entry.action != action:
                    continue
                if subject and entry.subject != subject:
                    continue
                if resource and entry.resource != resource:
                    continue
                if since and entry.timestamp < since:
                    continue
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain."""
        with self._lock:
            prev = ""
            for entry in self._entries:
                if entry.prev_hash != prev:
                    return False
                computed = self._compute_hash(entry)
                if entry.hash != computed:
                    return False
                prev = entry.hash
        return True

    @staticmethod
    def _compute_hash(entry: AuditEntry) -> str:
        """Recompute hash of an entry."""
        data = json.dumps({
            "action": entry.action.value,
            "subject": entry.subject,
            "resource": entry.resource,
            "timestamp": entry.timestamp,
            "success": entry.success,
            "prev_hash": entry.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_recent(self, n: int = 10) -> list[AuditEntry]:
        """Get the N most recent entries."""
        with self._lock:
            return list(self._entries)[-n:]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._last_hash = ""
            self._counter = 0
