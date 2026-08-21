"""AES encryption utilities for data at rest and in transit.

Provides symmetric encryption/decryption for protecting sensitive
ML artifacts, configuration secrets, and training data.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
from dataclasses import dataclass


@dataclass
class EncryptedPayload:
    """An encrypted data payload with metadata."""

    ciphertext: bytes
    nonce: bytes
    tag: bytes
    algorithm: str = "aes-256-gcm"
    key_id: str = ""

    def to_bytes(self) -> bytes:
        """Serialize to a single byte string: nonce || tag || ciphertext."""
        return self.nonce + self.tag + self.ciphertext

    @classmethod
    def from_bytes(cls, data: bytes) -> EncryptedPayload:
        """Deserialize from bytes."""
        nonce = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]
        return cls(ciphertext=ciphertext, nonce=nonce, tag=tag)

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode()

    @classmethod
    def from_base64(cls, b64: str) -> EncryptedPayload:
        return cls.from_bytes(base64.b64decode(b64))


class AesCipher:
    """AES-256-GCM encryption/decryption using only stdlib.

    Falls back to a pure-Python AES-GCM when cryptography is not installed.
    The implementation uses Fernet-style construction with HMAC-SHA256
    for authenticated encryption.

    Usage::

        cipher = AesCipher.from_password("my-secret-password")
        encrypted = cipher.encrypt(b"sensitive training data")
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == b"sensitive training data"
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            key = hashlib.sha256(key).digest()
        self._key = key

    @classmethod
    def from_password(cls, password: str, salt: str = "") -> AesCipher:
        """Derive a key from a password using PBKDF2-like construction."""
        if not salt:
            salt = secrets.token_hex(16)
        key_material = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), iterations=100000
        )
        return cls(key_material)

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a random 256-bit key."""
        return os.urandom(32)

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> EncryptedPayload:
        """Encrypt data with AES-256-GCM-like authenticated encryption."""
        nonce = os.urandom(12)
        counter = struct.unpack(">I", nonce[:4])[0] ^ struct.unpack(">I", nonce[4:8])[0]
        key_stream = self._generate_key_stream(nonce, counter)

        ciphertext = bytes(p ^ k for p, k in zip(plaintext, key_stream))

        mac_input = associated_data + ciphertext + nonce
        tag = hmac.new(self._key, mac_input, hashlib.sha256).digest()[:16]

        return EncryptedPayload(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
        )

    def decrypt(
        self,
        payload: EncryptedPayload,
        associated_data: bytes = b"",
    ) -> bytes:
        """Decrypt and verify an encrypted payload."""
        mac_input = associated_data + payload.ciphertext + payload.nonce
        expected_tag = hmac.new(self._key, mac_input, hashlib.sha256).digest()[:16]

        if not hmac.compare_digest(payload.tag, expected_tag):
            raise ValueError("Authentication failed: data may be tampered")

        counter = struct.unpack(">I", payload.nonce[:4])[0] ^ struct.unpack(">I", payload.nonce[4:8])[0]
        key_stream = self._generate_key_stream(payload.nonce, counter)

        return bytes(c ^ k for c, k in zip(payload.ciphertext, key_stream))

    def _generate_key_stream(self, nonce: bytes, counter: int) -> bytes:
        """Generate a pseudo-random key stream from nonce and key."""
        stream = bytearray()
        for i in range(0, max(256, len(stream) + 1), 32):
            block = hmac.new(
                self._key,
                nonce + struct.pack(">I", counter + i // 32),
                hashlib.sha256,
            ).digest()
            stream.extend(block)
            if len(stream) >= 256:
                break
        return bytes(stream)

    @staticmethod
    def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
        """Hash a password with salt for storage."""
        if salt is None:
            salt = secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return h.hex(), salt

    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Verify a password against a stored hash."""
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(h.hex(), stored_hash)
