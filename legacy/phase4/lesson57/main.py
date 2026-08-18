# Silverwing ML
# Phase 4 - Lesson 57
# Dynamic AI Model and Provider Routing


import json
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 57")
print("Dynamic AI Model and Provider Routing")
print()


# ==================================================
# 1. UTILITIES
# ==================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. MODEL RECORD
# ==================================================

@dataclass
class ModelRecord:

    provider: str

    model_name: str

    model_type: str

    capabilities: List[str]

    context_length: int

    local: bool = False

    available: bool = True

    latency_ms: float = 100.0

    cost_per_request: float = 0.0

    quality_score: float = 0.5

    metadata: Dict = field(
        default_factory=dict
    )


    def to_dict(self):

        return {
            "provider":
                self.provider,

            "model_name":
                self.model_name,

            "model_type":
                self.model_type,

            "capabilities":
                self.capabilities,

            "context_length":
                self.context_length,

            "local":
                self.local,

            "available":
                self.available,

            "latency_ms":
                self.latency_ms,

            "cost_per_request":
                self.cost_per_request,

            "quality_score":
                self.quality_score,

            "metadata":
                self.metadata
        }


# ==================================================
# 3. MODEL REGISTRY
# ==================================================

class ModelRegistry:

    def __init__(self):

        self.models: Dict[
            str,
            ModelRecord
        ] = {}


    def register(
            self,
            model: ModelRecord
    ):

        key = (
            f"{model.provider}:"
            f"{model.model_name}"
        )


        self.models[key] = model


    def get(
            self,
            provider: str,
            model_name: str
    ):

        key = (
            f"{provider}:"
            f"{model_name}"
        )


        return self.models.get(
            key
        )


    def list_models(self):

        return list(
            self.models.values()
        )


    def available_models(self):

        return [
            model
            for model
            in self.list_models()
            if model.available
        ]


    def find_capability(
            self,
            capability: str
    ):

        return [
            model
            for model
            in self.available_models()
            if capability
               in model.capabilities
        ]


registry = ModelRegistry()


# ==================================================
# 4. REGISTER EXAMPLE MODELS
# ==================================================

print("TEST 1: Model Registry")
print()


registry.register(
    ModelRecord(
        provider="local",
        model_name="tiny-gpt2",
        model_type="text",
        capabilities=[
            "text_generation",
            "completion"
        ],
        context_length=1024,
        local=True,
        available=True,
        latency_ms=80,
        cost_per_request=0.0,
        quality_score=0.25,
        metadata={
            "purpose":
                "local development model"
        }
    )
)


registry.register(
    ModelRecord(
        provider="local",
        model_name="silverwing-ml",
        model_type="machine_learning",
        capabilities=[
            "machine_prediction",
            "classification"
        ],
        context_length=0,
        local=True,
        available=True,
        latency_ms=35,
        cost_per_request=0.0,
        quality_score=0.80,
        metadata={
            "purpose":
                "machine risk prediction"
        }
    )
)


registry.register(
    ModelRecord(
        provider="cloud",
        model_name="general-chat-model",
        model_type="text",
        capabilities=[
            "text_generation",
            "reasoning",
            "tool_calling"
        ],
        context_length=32768,
        local=False,
        available=True,
        latency_ms=450,
        cost_per_request=0.01,
        quality_score=0.85,
        metadata={
            "purpose":
                "general AI reasoning"
        }
    )
)


registry.register(
    ModelRecord(
        provider="specialist",
        model_name="vision-model",
        model_type="vision",
        capabilities=[
            "image_analysis",
            "vision"
        ],
        context_length=16384,
        local=False,
        available=True,
        latency_ms=500,
        cost_per_request=0.02,
        quality_score=0.90,
        metadata={
            "purpose":
                "image understanding"
        }
    )
)


for model in registry.list_models():

    print(
        model.provider,
        "->",
        model.model_name
    )

print()


# ==================================================
# 5. DISPLAY MODELS
# ==================================================

print("TEST 2: Model Inventory")
print()


for model in registry.list_models():

    print(
        json.dumps(
            model.to_dict(),
            indent=4
        )
    )

    print()


# ==================================================
# 6. TASK REQUEST
# ==================================================

@dataclass
class ModelRequest:

    task: str

    required_capability: str

    minimum_quality: float = 0.0

    maximum_latency_ms: Optional[float] = None

    maximum_cost: Optional[float] = None

    minimum_context_length: int = 0

    prefer_local: bool = False

    prefer_fast: bool = False


print("TEST 3: Model Request")
print()


request = ModelRequest(
    task="Answer a technical question.",
    required_capability="reasoning",
    minimum_quality=0.70,
    maximum_latency_ms=1000,
    maximum_cost=0.05
)


print(
    request
)

print()


# ==================================================
# 7. FILTER MODELS
# ==================================================

def filter_models(
        registry: ModelRegistry,
        request: ModelRequest
):

    candidates = []


    for model in (
            registry.available_models()
    ):

        if request.required_capability not in (
                model.capabilities
        ):

            continue


        if (
                model.quality_score
                <
                request.minimum_quality
        ):

            continue


        if (
                request.maximum_latency_ms
                is not None
                and
                model.latency_ms
                >
                request.maximum_latency_ms
        ):

            continue


        if (
                request.maximum_cost
                is not None
                and
                model.cost_per_request
                >
                request.maximum_cost
        ):

            continue


        if (
                model.context_length
                <
                request.minimum_context_length
        ):

            continue


        candidates.append(
            model
        )


    return candidates


print("TEST 4: Candidate Filtering")
print()


candidates = filter_models(
    registry,
    request
)


for model in candidates:

    print(
        model.provider,
        model.model_name
    )

print()


# ==================================================
# 8. MODEL SCORING
# ==================================================

def score_model(
        model: ModelRecord,
        request: ModelRequest
):

    score = 0.0


    # Quality contribution.

    score += (
            model.quality_score
            *
            50
    )


    # Latency contribution.

    latency_score = (
            1.0
            /
            max(
                model.latency_ms,
                1
            )
    )


    score += (
            latency_score
            *
            1000
    )


    # Cost contribution.

    if model.cost_per_request == 0:

        score += 20

    else:

        score += (
                         1.0
                         /
                         model.cost_per_request
                 ) * 0.01


    # Local preference.

    if (
            request.prefer_local
            and
            model.local
    ):

        score += 30


    # Fast preference.

    if request.prefer_fast:

        score += (
                         1.0
                         /
                         max(
                             model.latency_ms,
                             1
                         )
                 ) * 500


    return score


print("TEST 5: Model Scoring")
print()


for model in candidates:

    print(
        model.model_name,
        "score:",
        round(
            score_model(
                model,
                request
            ),
            4
        )
    )

print()


# ==================================================
# 9. ROUTER
# ==================================================

class ModelRouter:

    def __init__(
            self,
            registry: ModelRegistry
    ):

        self.registry = registry


    def route(
            self,
            request: ModelRequest
    ):

        candidates = filter_models(
            self.registry,
            request
        )


        if not candidates:

            return None


        ranked = sorted(
            candidates,
            key=lambda model:
            score_model(
                model,
                request
            ),
            reverse=True
        )


        return ranked[0]


router = ModelRouter(
    registry
)


# ==================================================
# 10. ROUTE REQUEST
# ==================================================

print("TEST 6: Route AI Request")
print()


selected = router.route(
    request
)


if selected:

    print(
        "Selected provider:",
        selected.provider
    )

    print(
        "Selected model:",
        selected.model_name
    )

else:

    print(
        "No suitable model found."
    )


print()


# ==================================================
# 11. DIFFERENT TASK TYPES
# ==================================================

print("TEST 7: Capability-Based Routing")
print()


requests_to_test = [

    ModelRequest(
        task="Answer a technical question.",
        required_capability="reasoning",
        minimum_quality=0.70
    ),

    ModelRequest(
        task="Analyze an image.",
        required_capability="vision",
        minimum_quality=0.70
    ),

    ModelRequest(
        task="Predict machine risk.",
        required_capability="machine_prediction"
    ),

    ModelRequest(
        task="Complete text locally.",
        required_capability="text_generation",
        prefer_local=True,
        maximum_cost=0.0
    )
]


for task_request in requests_to_test:

    selected_model = router.route(
        task_request
    )


    print(
        "Task:",
        task_request.task
    )


    if selected_model:

        print(
            "Provider:",
            selected_model.provider
        )

        print(
            "Model:",
            selected_model.model_name
        )

    else:

        print(
            "No suitable model."
        )


    print()


# ==================================================
# 12. LOCAL-FIRST ROUTING
# ==================================================

print("TEST 8: Local-First Routing")
print()


local_request = ModelRequest(
    task="Generate a response privately.",
    required_capability="text_generation",
    prefer_local=True,
    minimum_quality=0.1
)


local_model = router.route(
    local_request
)


if local_model:

    print(
        "Selected:",
        local_model.provider,
        local_model.model_name
    )

else:

    print(
        "No local-compatible model."
    )

print()


# ==================================================
# 13. FASTEST MODEL
# ==================================================

print("TEST 9: Fast Routing")
print()


fast_request = ModelRequest(
    task="Respond quickly.",
    required_capability="text_generation",
    minimum_quality=0.1,
    prefer_fast=True
)


fast_model = router.route(
    fast_request
)


if fast_model:

    print(
        "Fast route:",
        fast_model.provider,
        fast_model.model_name
    )

print()


# ==================================================
# 14. MODEL UNAVAILABLE
# ==================================================

print("TEST 10: Provider Failure")
print()


cloud_model = registry.get(
    "cloud",
    "general-chat-model"
)


cloud_model.available = False


failure_request = ModelRequest(
    task="Reason about an architecture problem.",
    required_capability="reasoning",
    minimum_quality=0.70
)


fallback_model = router.route(
    failure_request
)


if fallback_model:

    print(
        "Fallback model:",
        fallback_model.provider,
        fallback_model.model_name
    )

else:

    print(
        "No fallback model available."
    )


cloud_model.available = True


print()


# ==================================================
# 15. FALLBACK CHAIN
# ==================================================

print("TEST 11: Fallback Chain")
print()


class FallbackRouter:

    def __init__(
            self,
            registry
    ):

        self.registry = registry


    def route_chain(
            self,
            request
    ):

        candidates = filter_models(
            self.registry,
            request
        )


        ranked = sorted(
            candidates,
            key=lambda model:
            score_model(
                model,
                request
            ),
            reverse=True
        )


        return ranked


fallback_router = FallbackRouter(
    registry
)


chain = fallback_router.route_chain(
    ModelRequest(
        task="General reasoning",
        required_capability="reasoning",
        minimum_quality=0.1
    )
)


for index, model in enumerate(
        chain,
        start=1
):

    print(
        index,
        "->",
        model.provider,
        model.model_name
    )

print()


# ==================================================
# 16. PROVIDER HEALTH
# ==================================================

print("TEST 12: Provider Health")
print()


def simulate_provider_health(
        model: ModelRecord
):

    if not model.available:

        return {
            "status":
                "offline",

            "latency_ms":
                None
        }


    return {
        "status":
            "healthy",

        "latency_ms":
            model.latency_ms
    }


for model in registry.list_models():

    health = simulate_provider_health(
        model
    )


    print(
        model.provider,
        model.model_name,
        "->",
        health
    )

print()


# ==================================================
# 17. MODEL PERFORMANCE UPDATE
# ==================================================

print("TEST 13: Dynamic Performance Update")
print()


def update_latency(
        model,
        observed_latency_ms
):

    model.latency_ms = (
            0.7 * model.latency_ms
            +
            0.3 * observed_latency_ms
    )


tracked_model = registry.get(
    "cloud",
    "general-chat-model"
)


print(
    "Original latency:",
    tracked_model.latency_ms
)


update_latency(
    tracked_model,
    300
)


print(
    "Updated latency:",
    round(
        tracked_model.latency_ms,
        2
    )
)

print()


# ==================================================
# 18. ROUTING DECISION RECORD
# ==================================================

print("TEST 14: Routing Decision")
print()


selected_model = router.route(
    request
)


routing_decision = {
    "request_id":
        str(uuid.uuid4()),

    "timestamp":
        utc_now(),

    "task":
        request.task,

    "required_capability":
        request.required_capability,

    "selected_provider":
        selected_model.provider
        if selected_model
        else None,

    "selected_model":
        selected_model.model_name
        if selected_model
        else None,

    "reason":
        "highest compatible routing score"
}


print(
    json.dumps(
        routing_decision,
        indent=4
    )
)

print()


# ==================================================
# 19. MODEL INVOCATION ABSTRACTION
# ==================================================

print("TEST 15: Model Provider Abstraction")
print()


class ModelProvider:

    def __init__(
            self,
            model_record
    ):

        self.model = model_record


    def generate(
            self,
            prompt: str
    ):

        # Educational placeholder.
        #
        # A real provider adapter will call
        # the actual local or remote model.

        return {
            "provider":
                self.model.provider,

            "model":
                self.model.model_name,

            "prompt":
                prompt,

            "status":
                "simulated_generation"
        }


provider = ModelProvider(
    selected_model
)


generation_result = provider.generate(
    "Explain machine learning."
)


print(
    json.dumps(
        generation_result,
        indent=4
    )
)

print()


# ==================================================
# 20. AI GATEWAY CONCEPT
# ==================================================

print("AI GATEWAY CONCEPT")
print()

print("Agent")
print("  ↓")
print("Model Router")
print("  ↓")
print("Provider Adapter")
print("  ↓")
print("┌──────────┬──────────┬──────────┐")
print("↓          ↓          ↓")
print("Local     Cloud     Specialist")
print("Model     Model       Model")
print("└──────────┴──────────┴──────────┘")
print("  ↓")
print("Generated Result")
print("  ↓")
print("Agent")

print()


# ==================================================
# 21. WHY MODEL ROUTING MATTERS
# ==================================================

print("WHY MODEL ROUTING MATTERS")
print()

print(
    "Different tasks can require different "
    "models."
)

print()

print(
    "A local model can be useful for privacy, "
    "low cost, or offline operation."
)

print()

print(
    "A stronger remote model may be useful "
    "for difficult reasoning tasks."
)

print()

print(
    "A specialist model may be better for "
    "vision, speech, coding, embeddings, or "
    "machine-learning prediction."
)

print()


# ==================================================
# 22. IMPORTANT ARCHITECTURE PRINCIPLE
# ==================================================

print("ARCHITECTURE PRINCIPLE")
print()

print(
    "Silverwing should not permanently depend "
    "on one model vendor."
)

print()

print(
    "Provider adapters should expose a common "
    "interface to the routing layer."
)

print()

print(
    "The router chooses an available model "
    "according to the task."
)

print()


# ==================================================
# 23. FUTURE MODEL CAPABILITIES
# ==================================================

print("FUTURE MODEL CAPABILITIES")
print()

capabilities = [
    "text_generation",
    "reasoning",
    "tool_calling",
    "code_generation",
    "vision",
    "speech_to_text",
    "text_to_speech",
    "embeddings",
    "classification",
    "machine_prediction",
    "summarization",
    "translation"
]


for capability in capabilities:

    print(
        "-",
        capability
    )

print()


# ==================================================
# 24. SILVERWING MODEL ARCHITECTURE
# ==================================================

print("SILVERWING MODEL ARCHITECTURE")
print()

print("                    Agent")
print("                      ↓")
print("                Model Router")
print("                      ↓")
print("              Capability Match")
print("                      ↓")
print("             Availability Check")
print("                      ↓")
print("             Quality / Latency")
print("                      ↓")
print("                 Cost Check")
print("                      ↓")
print("              Provider Adapter")
print("                      ↓")
print("              Selected Model")
print("                      ↓")
print("                  Result")
print("                      ↓")
print("                    Agent")

print()


# ==================================================
# 25. ADVANCED PERSONAL AI
# ==================================================

print("ADVANCED PERSONAL AI")
print()

print(
    "This allows Silverwing to select different "
    "AI capabilities for different operations "
    "without changing the agent itself."
)

print()

print(
    "That is the foundation for a modular model "
    "ecosystem rather than a single-model application."
)

print()


# ==================================================
# 26. CURRENT SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("LLM")
print(" ↓")
print("Conversation")
print(" ↓")
print("Memory")
print(" ↓")
print("Semantic Retrieval")
print(" ↓")
print("Tools")
print(" ↓")
print("Planning")
print(" ↓")
print("Multitasking")
print(" ↓")
print("Verification")
print(" ↓")
print("Persistent Jobs")
print(" ↓")
print("Service Communication")
print(" ↓")
print("Service Discovery")
print(" ↓")
print("Capability Routing")
print(" ↓")
print("Dynamic Model Routing")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 57 COMPLETE ===")