"""JWT-like token system for stateless authentication.

Provides token creation, validation, refresh, and claims management
using HMAC-SHA256 signing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenClaims:
    """Claims embedded in a token."""

    subject: str
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    issuer: str = "silverwing"
    audience: str = "api"
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_id: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    @property
    def expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        if self.expires_at <= 0:
            return float("inf")
        return max(0, self.expires_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.subject,
            "iat": self.issued_at,
            "exp": self.expires_at,
            "iss": self.issuer,
            "aud": self.audience,
            "scopes": self.scopes,
            "jti": self.token_id,
            "meta": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenClaims:
        return cls(
            subject=data.get("sub", ""),
            issued_at=data.get("iat", 0),
            expires_at=data.get("exp", 0),
            issuer=data.get("iss", ""),
            audience=data.get("aud", ""),
            scopes=data.get("scopes", []),
            token_id=data.get("jti", ""),
            metadata=data.get("meta", {}),
        )


class TokenService:
    """JWT-like token service with HMAC-SHA256 signing.

    Usage::

        service = TokenService(secret="my-secret-key")
        token = service.create_token(
            subject="user-1",
            scopes=["read", "write"],
            ttl_seconds=3600,
        )

        claims = service.validate_token(token)
        if claims and not claims.expired:
            print(f"Authenticated: {claims.subject}")
    """

    def __init__(self, secret: str | None = None, algorithm: str = "HS256") -> None:
        self._secret = (secret or secrets.token_hex(32)).encode()
        self._algorithm = algorithm

    def create_token(
        self,
        subject: str,
        scopes: list[str] | None = None,
        ttl_seconds: float = 3600.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a signed token string."""
        now = time.time()
        claims = TokenClaims(
            subject=subject,
            issued_at=now,
            expires_at=now + ttl_seconds,
            scopes=scopes or [],
            metadata=metadata or {},
        )
        return self._encode(claims)

    def validate_token(self, token: str) -> TokenClaims | None:
        """Validate and decode a token. Returns None if invalid."""
        try:
            claims = self._decode(token)
            if claims is None:
                return None
            if claims.expired:
                return None
            return claims
        except Exception:
            return None

    def refresh_token(
        self,
        token: str,
        ttl_seconds: float = 3600.0,
    ) -> str | None:
        """Refresh a token by re-issuing with new expiry."""
        claims = self.validate_token(token)
        if claims is None:
            return None
        return self.create_token(
            subject=claims.subject,
            scopes=claims.scopes,
            ttl_seconds=ttl_seconds,
            metadata=claims.metadata,
        )

    def has_scope(self, token: str, scope: str) -> bool:
        """Check if a token has a specific scope."""
        claims = self.validate_token(token)
        if claims is None:
            return False
        return scope in claims.scopes or "*" in claims.scopes

    def _encode(self, claims: TokenClaims) -> str:
        payload = json.dumps(claims.to_dict(), separators=(",", ":"))
        payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        signature = self._sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def _decode(self, token: str) -> TokenClaims | None:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        expected = self._sign(payload_b64)
        if not hmac.compare_digest(signature, expected):
            return None
        padding = 4 - len(payload_b64) % 4
        payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return TokenClaims.from_dict(payload)

    def _sign(self, payload_b64: str) -> str:
        sig = hmac.new(self._secret, payload_b64.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).decode().rstrip("=")
