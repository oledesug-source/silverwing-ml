"""Role-Based Access Control (RBAC) for ML serving endpoints.

Provides role/permission management, policy evaluation, and
role hierarchy for fine-grained authorization.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Permission(enum.Enum):
    """Granular permissions for ML operations."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    TRAIN = "train"
    DEPLOY = "deploy"
    ADMIN = "admin"
    DELETE = "delete"
    BENCHMARK = "benchmark"
    EXPORT = "export"


@dataclass
class Role:
    """A named role with a set of permissions."""

    name: str
    permissions: set[Permission] = field(default_factory=set)
    inherits_from: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Subject:
    """A user or service account."""

    subject_id: str
    name: str = ""
    roles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RBAC:
    """Role-Based Access Control system.

    Usage::

        rbac = RBAC()
        rbac.add_role(Role(
            name="researcher",
            permissions={Permission.READ, Permission.TRAIN, Permission.BENCHMARK},
        ))
        rbac.add_role(Role(
            name="admin",
            permissions={Permission.ADMIN},
            inherits_from=["researcher"],
        ))

        subject = Subject(subject_id="user-1", roles=["researcher"])
        assert rbac.has_permission(subject, Permission.READ)
        assert not rbac.has_permission(subject, Permission.DELETE)
    """

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._hierarchy: dict[str, set[Permission]] = {}

    def add_role(self, role: Role) -> None:
        """Register a role."""
        self._roles[role.name] = role
        self._hierarchy.pop(role.name, None)

    def remove_role(self, name: str) -> bool:
        """Remove a role."""
        if name in self._roles:
            del self._roles[name]
            self._hierarchy.pop(name, None)
            return True
        return False

    def get_role(self, name: str) -> Role | None:
        return self._roles.get(name)

    def list_roles(self) -> list[Role]:
        return list(self._roles.values())

    def _resolve_permissions(self, role_name: str, visited: set[str] | None = None) -> set[Permission]:
        """Resolve all permissions for a role including inherited ones."""
        if role_name in (visited or set()):
            return set()
        visited = visited or set()
        visited.add(role_name)

        if role_name in self._hierarchy:
            return self._hierarchy[role_name]

        role = self._roles.get(role_name)
        if role is None:
            return set()

        perms = set(role.permissions)
        for parent_name in role.inherits_from:
            perms |= self._resolve_permissions(parent_name, visited)

        self._hierarchy[role_name] = perms
        return perms

    def get_permissions(self, subject: Subject) -> set[Permission]:
        """Get all effective permissions for a subject."""
        perms: set[Permission] = set()
        for role_name in subject.roles:
            perms |= self._resolve_permissions(role_name)
        return perms

    def has_permission(self, subject: Subject, permission: Permission) -> bool:
        """Check if a subject has a specific permission."""
        return permission in self.get_permissions(subject)

    def has_any_permission(
        self, subject: Subject, permissions: set[Permission]
    ) -> bool:
        """Check if a subject has any of the given permissions."""
        subject_perms = self.get_permissions(subject)
        return bool(subject_perms & permissions)

    def require_permission(
        self, subject: Subject, permission: Permission
    ) -> None:
        """Raise PermissionError if subject lacks the permission."""
        if not self.has_permission(subject, permission):
            raise PermissionError(
                f"Subject '{subject.subject_id}' lacks permission '{permission.value}'"
            )
