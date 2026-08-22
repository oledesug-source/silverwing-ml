"""SilverWing Platform — Layer 4 abstractions.

Provides the highest-level platform services: database persistence,
model providers, policy engine, and approval management.

Named ``silverwing_platform`` to avoid collision with Python's stdlib
``platform`` module.
"""

from silverwing_platform.approvals import ApprovalManager, ApprovalRequest, ApprovalStatus
from silverwing_platform.database import PlatformDatabase
from silverwing_platform.models import (
    GenerationConfig,
    GeneratorProvider,
    InferenceRequest,
    InferenceResponse,
    MockProvider,
    ModelMetadata,
    ModelProvider,
    ModelProviderError,
)
from silverwing_platform.policies import PolicyDecision, PolicyEngine, PolicyRule

__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalStatus",
    "GenerationConfig",
    "InferenceRequest",
    "InferenceResponse",
    "ModelMetadata",
    "ModelProvider",
    "ModelProviderError",
    "MockProvider",
    "PlatformDatabase",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyRule",
]
