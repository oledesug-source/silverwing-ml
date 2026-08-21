"""Internal compatibility helpers for foundation.ops.

Provides ``optional_dependency`` which returns the real module when present
or a lightweight no-op stub when absent, so callers can always ``import``
the name without ``ImportError``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _noop(*_args: Any, **_kwargs: Any) -> Any:
    return None


class _NoOpStub(ModuleType):
    """Module-like stub that silently ignores all attribute access."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _noop

    def __bool__(self) -> bool:
        return False


def optional_dependency(name: str) -> ModuleType:
    """Return the real module or a no-op stub when not installed.

    Used by foundation.ops trackers so their modules always import cleanly.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except Exception:
        return _NoOpStub(name)  # type: ignore[return-value]
