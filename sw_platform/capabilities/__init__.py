"""Capability schema, registry, and discovery."""

from .discovery import CapabilityDiscovery
from .registry import CapabilityRegistry
from .schema import CapabilitySchema

__all__ = ["CapabilitySchema", "CapabilityRegistry", "CapabilityDiscovery"]
