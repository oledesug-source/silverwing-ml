"""Authentication primitives for ML serving and API access.

Provides API key management, token generation, and HMAC-based
request signing.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field


@dataclass
class ApiKey:
    """An API key with metadata."""

    key_id: str
    key_hash: str
    name: str = ""
    scopes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    revoked: bool = False

    @property
    def valid(self) -> bool:
        """Check if the key is currently valid."""
        if self.revoked:
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True


@dataclass
class Token:
    """An authentication token."""

    token_id: str
    subject: str
    scopes: list[str] = field(default_factory=list)
    issued_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    @property
    def valid(self) -> bool:
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True


class Authenticator:
    """Manages API keys and validates requests.

    Usage::

        auth = Authenticator()
        key = auth.create_key("my-app", scopes=["read", "write"])
        print(key.key_id)

        # Validate a request
        if auth.validate_key(key.key_id, raw_key):
            print("Authorized")
    """

    def __init__(self) -> None:
        self._keys: dict[str, ApiKey] = {}
        self._tokens: dict[str, Token] = {}
        self._hmac_secret = secrets.token_bytes(32)

    def create_key(
        self,
        name: str,
        scopes: list[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> tuple[ApiKey, str]:
        """Create a new API key. Returns (metadata, raw_key)."""
        raw_key = secrets.token_urlsafe(32)
        key_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + ttl_seconds

        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or [],
            expires_at=expires_at,
        )
        self._keys[key_id] = api_key
        return api_key, raw_key

    def validate_key(self, key_id: str, raw_key: str) -> bool:
        """Validate a raw key against stored hash."""
        api_key = self._keys.get(key_id)
        if api_key is None or not api_key.valid:
            return False
        provided_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return hmac.compare_digest(provided_hash, api_key.key_hash)

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        api_key = self._keys.get(key_id)
        if api_key is None:
            return False
        api_key.revoked = True
        return True

    def create_token(
        self,
        subject: str,
        scopes: list[str] | None = None,
        ttl_seconds: float = 3600.0,
    ) -> Token:
        """Create an authentication token."""
        token_id = secrets.token_urlsafe(32)
        token = Token(
            token_id=token_id,
            subject=subject,
            scopes=scopes or [],
            expires_at=time.time() + ttl_seconds,
        )
        self._tokens[token_id] = token
        return token

    def validate_token(self, token_id: str) -> Token | None:
        """Validate a token. Returns the Token if valid, else None."""
        token = self._tokens.get(token_id)
        if token and token.valid:
            return token
        return None

    def sign_request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
    ) -> str:
        """Sign an HTTP request with HMAC. Returns the signature."""
        message = f"{method.upper()}\n{path}\n".encode() + body
        return hmac.new(
            self._hmac_secret, message, hashlib.sha256
        ).hexdigest()

    def verify_signature(
        self,
        method: str,
        path: str,
        body: bytes,
        signature: str,
    ) -> bool:
        """Verify an HMAC signature."""
        expected = self.sign_request(method, path, body)
        return hmac.compare_digest(expected, signature)

    def list_keys(self) -> list[ApiKey]:
        """List all API keys."""
        return list(self._keys.values())
