"""
Authentication, authorization, JWT, OAuth2, and RBAC.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "User",
    "AuthProvider",
    "JWT",
    "OAuth2Handler",
    "RBAC",
]


@dataclass
class User:
    """Represents an authenticated user with identity and role information."""

    id: str
    username: str
    email: str
    password_hash: str
    roles: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    is_active: bool = True


class AuthProvider:
    """User authentication and session management using password hashing and token-based sessions."""

    def __init__(self, session_expiry: int = 3600, iterations: int = 100_000) -> None:
        self._users: dict[str, User] = {}
        self._sessions: dict[str, tuple[User, float]] = {}
        self._iterations = iterations
        self._session_expiry = session_expiry

    def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user with a hashed password."""
        if any(u.username == username for u in self._users.values()):
            raise ValueError(f"Username '{username}' already exists")
        if any(u.email == email for u in self._users.values()):
            raise ValueError(f"Email '{email}' already exists")
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(password)
        user = User(id=user_id, username=username, email=email, password_hash=password_hash)
        self._users[user_id] = user
        return user

    def authenticate(self, username: str, password: str) -> User | None:
        """Authenticate with username and password, returning User or None."""
        for user in self._users.values():
            if user.username == username and user.is_active:
                if self.verify_password(password, user.password_hash):
                    return user
        return None

    def create_session(self, user: User) -> str:
        """Create a session token for the given user."""
        token = str(uuid.uuid4())
        self._sessions[token] = (user, time.time())
        return token

    def validate_session(self, token: str) -> User | None:
        """Validate a session token, returning User if valid and not expired."""
        if token not in self._sessions:
            return None
        user, created_at = self._sessions[token]
        if time.time() - created_at > self._session_expiry:
            del self._sessions[token]
            return None
        return user

    def destroy_session(self, token: str) -> None:
        """Destroy an active session."""
        self._sessions.pop(token, None)

    def hash_password(self, password: str) -> str:
        """Hash a password using PBKDF2-SHA256 with random salt."""
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, self._iterations)
        return f"pbkdf2:sha256:{self._iterations}:{salt.hex()}:{dk.hex()}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against its stored hash."""
        parts = stored_hash.split(":")
        if len(parts) != 5:
            return False
        scheme, algorithm, iterations_str, salt_hex, hash_hex = parts
        if scheme != "pbkdf2":
            return False
        salt = bytes.fromhex(salt_hex)
        iterations = int(iterations_str)
        dk = hashlib.pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk.hex(), hash_hex)

    def get_user_by_id(self, user_id: str) -> User | None:
        """Retrieve a user by their unique ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """Retrieve a user by their username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None


class JWT:
    """Simplified JSON Web Token implementation using base64 and HMAC."""

    @staticmethod
    def encode(payload: dict[str, Any], secret: str, algorithm: str = "HS256", expires_in: int = 3600) -> str:
        """Encode a payload into a JWT token string."""
        header = {"alg": algorithm, "typ": "JWT"}
        if "exp" not in payload:
            payload["exp"] = int(time.time()) + expires_in
        if "iat" not in payload:
            payload["iat"] = int(time.time())
        header_b64 = urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = urlsafe_b64encode(json.dumps(payload, default=str).encode()).decode().rstrip("=")
        signing_input = f"{header_b64}.{payload_b64}"
        if algorithm == "HS256":
            signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        elif algorithm == "HS512":
            signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha512).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        signature_b64 = urlsafe_b64encode(signature).decode().rstrip("=")
        return f"{header_b64}.{payload_b64}.{signature_b64}"

    @staticmethod
    def decode(token: str, secret: str, algorithm: str = "HS256") -> dict[str, Any]:
        """Decode and verify a JWT token, returning the payload."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token format")
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"
        if algorithm == "HS256":
            expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        elif algorithm == "HS512":
            expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha512).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        actual_sig = urlsafe_b64decode(signature_b64 + "==")
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid JWT signature")
        padding = 4 - len(payload_b64) % 4
        payload_b64_padded = payload_b64 + "=" * padding
        payload = json.loads(urlsafe_b64decode(payload_b64_padded))
        if "exp" in payload and payload["exp"] < time.time():
            raise ValueError("Token has expired")
        return payload


class OAuth2Handler:
    """Simulated OAuth2 authorization code flow handler."""

    def __init__(self, client_id: str, client_secret: str, auth_provider: AuthProvider, redirect_uri: str = "/oauth/callback") -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_provider = auth_provider
        self.redirect_uri = redirect_uri
        self._authorization_codes: dict[str, tuple[User, float]] = {}
        self._access_tokens: dict[str, tuple[User, float]] = {}

    def get_authorization_url(self, state: str = "") -> str:
        """Generate an authorization URL for the OAuth2 flow."""
        params = f"client_id={self.client_id}&redirect_uri={self.redirect_uri}&response_type=code"
        if state:
            params += f"&state={state}"
        return f"/oauth/authorize?{params}"

    def generate_authorization_code(self, user: User) -> str:
        """Generate an authorization code after user consent."""
        code = str(uuid.uuid4())
        self._authorization_codes[code] = (user, time.time())
        return code

    def exchange_code(self, code: str) -> str | None:
        """Exchange an authorization code for an access token."""
        if code not in self._authorization_codes:
            return None
        user, created_at = self._authorization_codes.pop(code)
        if time.time() - created_at > 600:
            return None
        token = str(uuid.uuid4())
        self._access_tokens[token] = (user, time.time())
        return token

    def validate_token(self, token: str) -> User | None:
        """Validate an access token and return the associated user."""
        if token not in self._access_tokens:
            return None
        user, created_at = self._access_tokens[token]
        if time.time() - created_at > 3600:
            del self._access_tokens[token]
            return None
        return user

    def revoke_token(self, token: str) -> None:
        """Revoke an access token."""
        self._access_tokens.pop(token, None)


class RBAC:
    """Role-Based Access Control with permission mapping."""

    def __init__(self) -> None:
        self._roles: dict[str, set[str]] = {}
        self._role_hierarchies: dict[str, set[str]] = {}

    def define_role(self, role_name: str, permissions: list[str] | None = None) -> None:
        """Define a role with a set of permissions."""
        self._roles[role_name] = set(permissions or [])

    def add_permission(self, role_name: str, permission: str) -> None:
        """Add a single permission to a role."""
        if role_name not in self._roles:
            self._roles[role_name] = set()
        self._roles[role_name].add(permission)

    def set_hierarchy(self, child_role: str, parent_role: str) -> None:
        """Set a role hierarchy where child inherits parent permissions."""
        if child_role not in self._role_hierarchies:
            self._role_hierarchies[child_role] = set()
        self._role_hierarchies[child_role].add(parent_role)

    def _get_effective_permissions(self, role_name: str, visited: set[str] | None = None) -> set[str]:
        """Get all permissions for a role including inherited ones."""
        if visited is None:
            visited = set()
        if role_name in visited:
            return set()
        visited.add(role_name)
        permissions = set(self._roles.get(role_name, set()))
        for parent in self._role_hierarchies.get(role_name, set()):
            permissions |= self._get_effective_permissions(parent, visited)
        return permissions

    def has_permission(self, user: User, permission: str) -> bool:
        """Check if a user has a specific permission through any of their roles."""
        for role in user.roles:
            effective = self._get_effective_permissions(role)
            if permission in effective:
                return True
        return False

    def has_role(self, user: User, role: str) -> bool:
        """Check if a user has a specific role."""
        return role in user.roles

    def require_permission(self, permission: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that checks for a specific permission before allowing access."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                user = kwargs.get("user") or (args[0] if args else None)
                if not isinstance(user, User):
                    raise PermissionError("Authentication required")
                if not self.has_permission(user, permission):
                    raise PermissionError(f"Permission '{permission}' denied for user '{user.username}'")
                return fn(*args, **kwargs)
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        return decorator

    def require_role(self, role: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that checks for a specific role before allowing access."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                user = kwargs.get("user") or (args[0] if args else None)
                if not isinstance(user, User):
                    raise PermissionError("Authentication required")
                if not self.has_role(user, role):
                    raise PermissionError(f"Role '{role}' required for user '{user.username}'")
                return fn(*args, **kwargs)
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        return decorator
