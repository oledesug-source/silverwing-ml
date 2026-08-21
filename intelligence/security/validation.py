"""Input validation and sanitization for ML serving endpoints.

Prevents injection attacks, validates payloads, and sanitizes user
input before it reaches model inference or database queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of input validation."""

    valid: bool
    errors: list[str]
    sanitized: Any = None

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors) if self.errors else ""


class InputValidator:
    """Validates structured inputs against schemas.

    Usage::

        validator = InputValidator()
        validator.add_rule("prompt", required=True, max_length=10000)
        validator.add_rule("temperature", required=False, min_val=0.0, max_val=2.0)

        result = validator.validate({"prompt": "Hello", "temperature": 0.7})
        if result.valid:
            process(result.sanitized)
    """

    def __init__(self) -> None:
        self._rules: dict[str, dict[str, Any]] = {}

    def add_rule(
        self,
        field: str,
        *,
        required: bool = True,
        max_length: int | None = None,
        min_length: int | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        pattern: str | None = None,
        allowed_types: tuple[type, ...] | None = None,
    ) -> None:
        """Add a validation rule for a field."""
        self._rules[field] = {
            "required": required,
            "max_length": max_length,
            "min_length": min_length,
            "min_val": min_val,
            "max_val": max_val,
            "pattern": pattern,
            "allowed_types": allowed_types,
        }

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate data against registered rules."""
        errors: list[str] = []
        sanitized: dict[str, Any] = {}

        for field, rules in self._rules.items():
            value = data.get(field)

            if value is None and rules["required"]:
                errors.append(f"'{field}' is required")
                continue

            if value is None:
                continue

            if rules["allowed_types"] and not isinstance(
                value, rules["allowed_types"]
            ):
                errors.append(
                    f"'{field}' must be {rules['allowed_types']}, got {type(value).__name__}"
                )
                continue

            if isinstance(value, str):
                if rules["max_length"] and len(value) > rules["max_length"]:
                    errors.append(f"'{field}' exceeds max length {rules['max_length']}")
                    continue
                if rules["min_length"] and len(value) < rules["min_length"]:
                    errors.append(f"'{field}' below min length {rules['min_length']}")
                    continue
                if rules["pattern"] and not re.match(rules["pattern"], value):
                    errors.append(f"'{field}' doesn't match pattern {rules['pattern']}")
                    continue

            if isinstance(value, (int, float)):
                if rules["min_val"] is not None and value < rules["min_val"]:
                    errors.append(f"'{field}' below minimum {rules['min_val']}")
                    continue
                if rules["max_val"] is not None and value > rules["max_val"]:
                    errors.append(f"'{field}' above maximum {rules['max_val']}")
                    continue

            sanitized[field] = value

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized=sanitized if not errors else None,
        )


class Sanitizer:
    """Sanitizes user input for safe processing.

    Usage::

        sanitizer = Sanitizer()
        clean = sanitizer.text(user_input)
        clean = sanitizer.filename(user_input)
    """

    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"\.\.\/",
        r"\b(exec|eval|import|system|subprocess)\b",
    ]

    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--|;|'|\")",
        r"(\bOR\b\s+\b\d+\b\s*=\s*\b\d+\b)",
    ]

    def text(self, value: str, max_length: int = 10000) -> str:
        """Sanitize general text input."""
        value = value[:max_length]
        for pattern in self.DANGEROUS_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        return value.strip()

    def filename(self, value: str) -> str:
        """Sanitize a filename."""
        value = value.split("/")[-1].split("\\")[-1]
        value = re.sub(r"[^\w\-.]", "_", value)
        value = re.sub(r"_+", "_", value)
        return value.strip("_") or "unnamed"

    def sql_input(self, value: str) -> ValidationResult:
        """Check for SQL injection patterns."""
        errors = []
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"Potential SQL injection detected: {pattern}")
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized=value,
        )

    def number(self, value: Any, min_val: float = 0.0, max_val: float = 1e10) -> float | None:
        """Validate and sanitize a numeric value."""
        try:
            num = float(value)
            if min_val <= num <= max_val:
                return num
            return None
        except (TypeError, ValueError):
            return None

    def strip_html(self, value: str) -> str:
        """Remove HTML tags from a string."""
        return re.sub(r"<[^>]+>", "", value)
