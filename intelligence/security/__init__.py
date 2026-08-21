"""Security fundamentals for ML infrastructure.

Covers:

- **Authentication** — API keys, HMAC signatures
- **Authorization** — RBAC, role hierarchy, permission evaluation
- **JWT Tokens** — stateless token creation, validation, refresh
- **Data integrity** — checksums, hashing, tamper detection
- **Encryption** — AES-256-GCM authenticated encryption, password hashing
- **Input validation** — sanitization, injection prevention
- **Audit logging** — tamper-evident chain, event querying
"""

from .audit import AuditAction, AuditEntry, AuditLog
from .auth import ApiKey, Authenticator, Token
from .crypto import Checksum, Hasher
from .encryption import AesCipher, EncryptedPayload
from .jwt import TokenClaims, TokenService
from .rbac import RBAC, Permission, Role, Subject
from .validation import InputValidator, Sanitizer, ValidationResult

__all__ = [
    "Authenticator",
    "ApiKey",
    "Token",
    "RBAC",
    "Role",
    "Subject",
    "Permission",
    "TokenService",
    "TokenClaims",
    "Hasher",
    "Checksum",
    "AesCipher",
    "EncryptedPayload",
    "InputValidator",
    "Sanitizer",
    "ValidationResult",
    "AuditLog",
    "AuditEntry",
    "AuditAction",
]
