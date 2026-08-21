"""Approval system (Phase 11).

When the policy engine returns ``require_approval`` for an action, an
:class:`ApprovalRequest` is created in ``pending`` status.  The action cannot
proceed until a human (or operator policy) marks it ``approved`` or
``rejected``.  Requests expire automatically after a configurable TTL.

Approval requests are persisted through :class:`PlatformDatabase` so they
survive restarts.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from silverwing_platform.database import PlatformDatabase


__all__ = [
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalManager",
]


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """A pending approval for a high-risk capability action."""

    request_id: str = field(default_factory=lambda: f"appr-{uuid.uuid4().hex[:12]}")
    user_id: str = ""
    session_id: str = ""
    capability_id: str = ""
    action: str = ""
    target: str = ""
    risk_level: str = "medium"
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "capability_id": self.capability_id,
            "action": self.action,
            "target": self.target,
            "risk_level": self.risk_level,
            "reason": self.reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "decision": self.decision,
        }


class ApprovalManager:
    """Manages approval requests, including persistence and TTL expiry.

    The LLM never bypasses approvals: :meth:`requires_approval` gates every
    action.  An approval is only granted when a prior ``approved`` decision
    exists for the request.
    """

    def __init__(
        self,
        db: Any = None,
        *,
        ttl_seconds: float = 300.0,
        auto_approve: bool = False,
    ) -> None:
        self._db = db
        self._ttl = ttl_seconds
        self._auto_approve = auto_approve
        self._pending: dict[str, ApprovalRequest] = {}

    @property
    def ttl(self) -> float:
        return self._ttl

    def request(
        self,
        capability_id: str,
        action: str,
        target: str,
        risk_level: str,
        reason: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> ApprovalRequest:
        """Create a new approval request in ``pending`` status."""
        req = ApprovalRequest(
            user_id=user_id,
            session_id=session_id,
            capability_id=capability_id,
            action=action,
            target=target,
            risk_level=risk_level,
            reason=reason,
            expires_at=time.time() + self._ttl,
        )
        self._pending[req.request_id] = req
        self._persist(req)
        return req

    def requires_approval(self, request_id: str) -> bool:
        """Return True if *request_id* is still pending (not yet approved).

        Expired requests are treated as requiring re-approval.  Never returns
        True for an already-approved request.
        """
        req = self.get(request_id)
        if req is None:
            return True
        if req.status in (ApprovalStatus.APPROVED,):
            return False
        if req.status == ApprovalStatus.EXPIRED:
            return True
        if req.status in (ApprovalStatus.REJECTED, ApprovalStatus.CANCELLED):
            return True
        if time.time() > req.expires_at:
            self._expire(req.request_id)
            return True
        return True

    def approve(self, request_id: str, decision: str = "") -> bool:
        """Approve a pending request. Returns True if found and pending."""
        req = self._pending.get(request_id)
        if req is None and self._db is not None:
            row = self._db.get_approval(request_id)
            if row:
                req = self._from_row(row)
        if req is None:
            return False
        if req.status != ApprovalStatus.PENDING:
            return False
        if time.time() > req.expires_at:
            self._expire(request_id)
            return False
        req.status = ApprovalStatus.APPROVED
        req.decision = decision or "approved by operator"
        self._pending[request_id] = req
        self._update(req)
        return True

    def reject(self, request_id: str, decision: str = "") -> bool:
        req = self.get(request_id)
        if req is None:
            return False
        req.status = ApprovalStatus.REJECTED
        req.decision = decision or "rejected by operator"
        self._pending[request_id] = req
        self._update(req)
        return True

    def cancel(self, request_id: str) -> bool:
        req = self.get(request_id)
        if req is None:
            return False
        req.status = ApprovalStatus.CANCELLED
        self._pending[request_id] = req
        self._update(req)
        return True

    def get(self, request_id: str) -> ApprovalRequest | None:
        req = self._pending.get(request_id)
        if req is not None:
            if time.time() > req.expires_at and req.status == ApprovalStatus.PENDING:
                self._expire(request_id)
                req = self._pending.get(request_id)
            return req
        if self._db is not None:
            row = self._db.get_approval(request_id)
            if row:
                return self._from_row(row)
        return None

    def list(self, status: str | None = None) -> list[ApprovalRequest]:
        if self._db is not None:
            rows = self._db.list_approvals(status)
            return [self._from_row(r) for r in rows]
        results = list(self._pending.values())
        if status:
            results = [r for r in results if r.status.value == status]
        return sorted(results, key=lambda r: r.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self, req: ApprovalRequest) -> None:
        if self._db is None:
            return
        try:
            from silverwing_platform.database import ApprovalRecord
            self._db.insert_approval(ApprovalRecord(
                request_id=req.request_id,
                capability_id=req.capability_id,
                action=req.action,
                target=req.target,
                risk_level=req.risk_level,
                reason=req.reason,
                user_id=req.user_id,
                session_id=req.session_id,
                status=req.status.value,
                created_at=req.created_at,
                expires_at=req.expires_at,
                decision=req.decision,
            ))
        except Exception as exc:
            logger.warning("Approval persistence failed: %s", exc)

    def _update(self, req: ApprovalRequest) -> None:
        if self._db is None:
            return
        self._db.update_approval(req.request_id, req.status.value, req.decision)

    def _from_row(self, row: dict[str, Any]) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=row["request_id"],
            user_id=row.get("user_id", ""),
            session_id=row.get("session_id", ""),
            capability_id=row.get("capability_id", ""),
            action=row.get("action", ""),
            target=row.get("target", ""),
            risk_level=row.get("risk_level", "medium"),
            reason=row.get("reason", ""),
            created_at=row.get("created_at") or time.time(),
            expires_at=row.get("expires_at") or 0.0,
            status=ApprovalStatus(row.get("status", "pending")) if row.get("status") else ApprovalStatus.PENDING,
            decision=row.get("decision", ""),
        )

    def _expire(self, request_id: str) -> None:
        req = self._pending.get(request_id)
        if req and req.status == ApprovalStatus.PENDING:
            req.status = ApprovalStatus.EXPIRED
            self._update(req)
