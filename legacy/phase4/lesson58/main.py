# Silverwing ML
# Phase 4 - Lesson 58
# Provider Adapters and Unified AI Gateway


import json
import time
import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 58")
print("Provider Adapters and Unified AI Gateway")
print()


# ==================================================
# 1. UTILITIES
# ==================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. AI REQUEST
# ==================================================

@dataclass
class AIRequest:

    prompt: str

    task_type: str = "text_generation"

    required_capabilities: List[str] = field(
        default_factory=lambda: [
            "text_generation"
        ]
    )

    temperature: float = 0.7

    max_tokens: int = 100

    prefer_local: bool = False

    maximum_cost: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 3. AI RESPONSE
# ==================================================

@dataclass
class AIResponse:

    provider: str

    model: str

    text: str

    usage: Dict[str, Any]

    latency_ms: float

    request_id: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 4. PROVIDER ADAPTER INTERFACE
# ==================================================

class ProviderAdapter(ABC):
    """
    Common interface implemented by every
    model provider.
    """

    @property
    @abstractmethod
    def provider_name(self):

        pass


    @abstractmethod
    def list_models(self):

        pass


    @abstractmethod
    def health(self):

        pass


    @abstractmethod
    def generate(
            self,
            request: AIRequest,
            model_name: str
    ):

        pass


# ==================================================
# 5. LOCAL PROVIDER
# ==================================================

class LocalProviderAdapter(
    ProviderAdapter
):
    """
    Educational local-model provider.

    In a production system this adapter could
    connect to Ollama, llama.cpp, vLLM,
    Transformers, or another local runtime.
    """

    @property
    def provider_name(self):

        return "local"


    def list_models(self):

        return [
            {
                "name":
                    "tiny-gpt2",

                "capabilities": [
                    "text_generation",
                    "completion"
                ],

                "local":
                    True,

                "cost":
                    0.0
            }
        ]


    def health(self):

        return {
            "provider":
                self.provider_name,

            "status":
                "healthy"
        }


    def generate(
            self,
            request: AIRequest,
            model_name: str
    ):

        start = time.perf_counter()


        # Educational response.
        #
        # Later this method will call the
        # actual local model.

        text = (
            f"[Local model {model_name}] "
            f"Response generated for: "
            f"{request.prompt}"
        )


        latency = (
                          time.perf_counter()
                          -
                          start
                  ) * 1000


        return AIResponse(
            provider=self.provider_name,
            model=model_name,
            text=text,
            usage={
                "prompt_tokens":
                    len(
                        request.prompt.split()
                    )
            },
            latency_ms=latency,
            request_id=str(
                uuid.uuid4()
            ),
            metadata={
                "execution":
                    "local"
            }
        )


# ==================================================
# 6. CLOUD PROVIDER
# ==================================================

class CloudProviderAdapter(
    ProviderAdapter
):
    """
    Educational cloud-provider adapter.

    No external API call is made in this lesson.
    """

    @property
    def provider_name(self):

        return "cloud"


    def list_models(self):

        return [
            {
                "name":
                    "general-chat-model",

                "capabilities": [
                    "text_generation",
                    "reasoning",
                    "tool_calling"
                ],

                "local":
                    False,

                "cost":
                    0.01
            },

            {
                "name":
                    "reasoning-model",

                "capabilities": [
                    "text_generation",
                    "reasoning"
                ],

                "local":
                    False,

                "cost":
                    0.03
            }
        ]


    def health(self):

        return {
            "provider":
                self.provider_name,

            "status":
                "healthy"
        }


    def generate(
            self,
            request: AIRequest,
            model_name: str
    ):

        start = time.perf_counter()


        text = (
            f"[Cloud model {model_name}] "
            f"Response generated for: "
            f"{request.prompt}"
        )


        latency = (
                          time.perf_counter()
                          -
                          start
                  ) * 1000


        return AIResponse(
            provider=self.provider_name,
            model=model_name,
            text=text,
            usage={
                "prompt_tokens":
                    len(
                        request.prompt.split()
                    )
            },
            latency_ms=latency,
            request_id=str(
                uuid.uuid4()
            ),
            metadata={
                "execution":
                    "cloud"
            }
        )


# ==================================================
# 7. SPECIALIST PROVIDER
# ==================================================

class SpecialistProviderAdapter(
    ProviderAdapter
):

    @property
    def provider_name(self):

        return "specialist"


    def list_models(self):

        return [
            {
                "name":
                    "vision-model",

                "capabilities": [
                    "vision",
                    "image_analysis"
                ],

                "local":
                    False,

                "cost":
                    0.02
            },

            {
                "name":
                    "embedding-model",

                "capabilities": [
                    "embeddings"
                ],

                "local":
                    True,

                "cost":
                    0.0
            }
        ]


    def health(self):

        return {
            "provider":
                self.provider_name,

            "status":
                "healthy"
        }


    def generate(
            self,
            request: AIRequest,
            model_name: str
    ):

        start = time.perf_counter()


        text = (
            f"[Specialist model {model_name}] "
            f"Processed task: "
            f"{request.prompt}"
        )


        latency = (
                          time.perf_counter()
                          -
                          start
                  ) * 1000


        return AIResponse(
            provider=self.provider_name,
            model=model_name,
            text=text,
            usage={
                "prompt_tokens":
                    len(
                        request.prompt.split()
                    )
            },
            latency_ms=latency,
            request_id=str(
                uuid.uuid4()
            ),
            metadata={
                "execution":
                    "specialist"
            }
        )


# ==================================================
# 8. PROVIDER REGISTRY
# ==================================================

class ProviderRegistry:

    def __init__(self):

        self.providers = {}


    def register(
            self,
            provider: ProviderAdapter
    ):

        self.providers[
            provider.provider_name
        ] = provider


    def get(
            self,
            provider_name
    ):

        return self.providers.get(
            provider_name
        )


    def list_providers(self):

        return list(
            self.providers.values()
        )


registry = ProviderRegistry()


registry.register(
    LocalProviderAdapter()
)

registry.register(
    CloudProviderAdapter()
)

registry.register(
    SpecialistProviderAdapter()
)


# ==================================================
# 9. PROVIDER INVENTORY
# ==================================================

print("TEST 1: Provider Inventory")
print()


for provider in (
        registry.list_providers()
):

    print(
        "Provider:",
        provider.provider_name
    )

    print(
        "Health:",
        provider.health()
    )

    print(
        "Models:"
    )

    for model in (
            provider.list_models()
    ):

        print(
            " ",
            model["name"]
        )

    print()


# ==================================================
# 10. MODEL CATALOG
# ==================================================

class ModelCatalog:

    def __init__(
            self,
            registry
    ):

        self.registry = registry


    def all_models(self):

        models = []


        for provider in (
                self.registry.list_providers()
        ):

            for model in (
                    provider.list_models()
            ):

                record = dict(
                    model
                )


                record["provider"] = (
                    provider.provider_name
                )


                models.append(
                    record
                )


        return models


    def find_capability(
            self,
            capability
    ):

        return [
            model
            for model in self.all_models()
            if capability
               in model["capabilities"]
        ]


catalog = ModelCatalog(
    registry
)


print("TEST 2: Unified Model Catalog")
print()


for model in (
        catalog.all_models()
):

    print(
        json.dumps(
            model,
            indent=4
        )
    )

print()


# ==================================================
# 11. ROUTING SCORE
# ==================================================

def routing_score(
        model,
        request
):

    score = 0.0


    # Capability match.

    score += 50


    # Local preference.

    if (
            request.prefer_local
            and
            model["local"]
    ):

        score += 30


    # Free model preference.

    if model["cost"] == 0:

        score += 20


    # Cost limit.

    if (
            request.maximum_cost
            is not None
            and
            model["cost"]
            <=
            request.maximum_cost
    ):

        score += 20


    return score


# ==================================================
# 12. MODEL ROUTER
# ==================================================

class GatewayRouter:

    def __init__(
            self,
            catalog
    ):

        self.catalog = catalog


    def select(
            self,
            request: AIRequest
    ):

        candidates = self.catalog.all_models()


        filtered = []


        for model in candidates:

            required = all(
                capability
                in model["capabilities"]
                for capability
                in request.required_capabilities
            )


            if not required:

                continue


            if (
                    request.maximum_cost
                    is not None
                    and
                    model["cost"]
                    >
                    request.maximum_cost
            ):

                continue


            if (
                    request.prefer_local
                    and
                    not model["local"]
            ):

                continue


            filtered.append(
                model
            )


        if not filtered:

            return None


        filtered.sort(
            key=lambda model:
            routing_score(
                model,
                request
            ),
            reverse=True
        )


        return filtered[0]


router = GatewayRouter(
    catalog
)


# ==================================================
# 13. ROUTER TEST
# ==================================================

print("TEST 3: Gateway Routing")
print()


requests_to_test = [

    AIRequest(
        prompt=(
            "Explain machine learning."
        ),
        required_capabilities=[
            "text_generation"
        ],
        prefer_local=True,
        maximum_cost=0.0
    ),

    AIRequest(
        prompt=(
            "Reason about a complex architecture."
        ),
        required_capabilities=[
            "reasoning"
        ],
        maximum_cost=0.05
    ),

    AIRequest(
        prompt=(
            "Analyze an image."
        ),
        required_capabilities=[
            "vision"
        ]
    ),

    AIRequest(
        prompt=(
            "Create embeddings for this document."
        ),
        required_capabilities=[
            "embeddings"
        ],
        prefer_local=True
    )
]


for request in requests_to_test:

    selected = router.select(
        request
    )


    print(
        "Task:",
        request.prompt
    )


    if selected:

        print(
            "Selected:",
            selected["provider"],
            selected["name"]
        )

    else:

        print(
            "No compatible model."
        )

    print()


# ==================================================
# 14. UNIFIED AI GATEWAY
# ==================================================

class AIGateway:

    def __init__(
            self,
            registry,
            router
    ):

        self.registry = registry

        self.router = router


    def generate(
            self,
            request: AIRequest
    ):

        selected = self.router.select(
            request
        )


        if selected is None:

            raise RuntimeError(
                "No compatible model available."
            )


        provider = self.registry.get(
            selected["provider"]
        )


        if provider is None:

            raise RuntimeError(
                "Provider adapter unavailable."
            )


        start = time.perf_counter()


        response = provider.generate(
            request,
            selected["name"]
        )


        total_latency = (
                                time.perf_counter()
                                -
                                start
                        ) * 1000


        response.metadata[
            "gateway"
        ] = "silverwing"


        response.metadata[
            "routing"
        ] = {
            "provider":
                selected["provider"],

            "model":
                selected["name"]
        }


        response.metadata[
            "gateway_latency_ms"
        ] = round(
            total_latency,
            3
        )


        return response


gateway = AIGateway(
    registry,
    router
)


# ==================================================
# 15. GATEWAY GENERATION
# ==================================================

print("TEST 4: Unified Gateway Generation")
print()


request = AIRequest(
    prompt=(
        "Explain how Silverwing uses AI models."
    ),
    required_capabilities=[
        "text_generation"
    ],
    prefer_local=True,
    maximum_cost=0.0
)


response = gateway.generate(
    request
)


print(
    json.dumps(
        {
            "provider":
                response.provider,

            "model":
                response.model,

            "text":
                response.text,

            "latency_ms":
                response.latency_ms,

            "request_id":
                response.request_id,

            "metadata":
                response.metadata
        },
        indent=4
    )
)

print()


# ==================================================
# 16. FALLBACK STRATEGY
# ==================================================

print("TEST 5: Fallback Strategy")
print()


class FallbackGateway:

    def __init__(
            self,
            registry,
            catalog
    ):

        self.registry = registry

        self.catalog = catalog


    def generate(
            self,
            request
    ):

        candidates = []


        for model in (
                self.catalog.all_models()
        ):

            required = all(
                capability
                in model["capabilities"]
                for capability
                in request.required_capabilities
            )


            if not required:

                continue


            if (
                    request.maximum_cost
                    is not None
                    and
                    model["cost"]
                    >
                    request.maximum_cost
            ):

                continue


            candidates.append(
                model
            )


        candidates.sort(
            key=lambda model:
            routing_score(
                model,
                request
            ),
            reverse=True
        )


        errors = []


        for candidate in candidates:

            provider = self.registry.get(
                candidate["provider"]
            )


            if provider is None:

                errors.append(
                    "Provider unavailable: "
                    +
                    candidate["provider"]
                )

                continue


            try:

                return provider.generate(
                    request,
                    candidate["name"]
                )

            except Exception as error:

                errors.append(
                    str(error)
                )


        raise RuntimeError(
            "All model candidates failed: "
            +
            "; ".join(
                errors
            )
        )


fallback_gateway = FallbackGateway(
    registry,
    catalog
)


fallback_response = (
    fallback_gateway.generate(
        AIRequest(
            prompt=(
                "Provide a general AI response."
            ),
            required_capabilities=[
                "text_generation"
            ]
        )
    )
)


print(
    "Fallback response:"
)

print(
    fallback_response.text
)

print()


# ==================================================
# 17. PROVIDER ADAPTER SEPARATION
# ==================================================

print("TEST 6: Provider Adapter Separation")
print()


print(
    "Agent interface:"
)

print(
    "gateway.generate(request)"
)

print()

print(
    "Provider-specific implementation:"
)

print(
    "provider.generate(request, model)"
)

print()

print(
    "The agent does not need to know "
    "which provider implementation was selected."
)

print()


# ==================================================
# 18. GATEWAY REQUEST RECORD
# ==================================================

print("TEST 7: Gateway Request Record")
print()


gateway_record = {
    "request_id":
        response.request_id,

    "timestamp":
        utc_now(),

    "task_type":
        request.task_type,

    "provider":
        response.provider,

    "model":
        response.model,

    "latency_ms":
        response.latency_ms,

    "status":
        "success"
}


print(
    json.dumps(
        gateway_record,
        indent=4
    )
)

print()


# ==================================================
# 19. MODEL FAILOVER SIMULATION
# ==================================================

print("TEST 8: Model Failover")
print()


local_provider = registry.get(
    "local"
)


original_models = list(
    local_provider.list_models()
)


# Simulate local text model becoming unavailable
# by temporarily changing routing behavior.

local_model_name = (
    "tiny-gpt2"
)


request_for_failover = AIRequest(
    prompt=(
        "Explain artificial intelligence."
    ),
    required_capabilities=[
        "text_generation"
    ]
)


print(
    "Primary candidates:"
)

for model in (
        catalog.find_capability(
            "text_generation"
        )
):

    print(
        "-",
        model["provider"],
        model["name"]
    )

print()


print(
    "Fallback mechanism is ready to choose "
    "another compatible provider when the "
    "preferred one is unavailable."
)

print()


# ==================================================
# 20. HEALTH AGGREGATION
# ==================================================

print("TEST 9: Provider Health")
print()


provider_health = {}


for provider in (
        registry.list_providers()
):

    provider_health[
        provider.provider_name
    ] = provider.health()


print(
    json.dumps(
        provider_health,
        indent=4
    )
)

print()


# ==================================================
# 21. CAPABILITY MATRIX
# ==================================================

print("TEST 10: Capability Matrix")
print()


capability_matrix = {}


for model in (
        catalog.all_models()
):

    for capability in (
            model["capabilities"]
    ):

        if capability not in (
                capability_matrix
        ):

            capability_matrix[
                capability
            ] = []


        capability_matrix[
            capability
        ].append(
            (
                model["provider"],
                model["name"]
            )
        )


print(
    json.dumps(
        capability_matrix,
        indent=4
    )
)

print()


# ==================================================
# 22. UNIFIED AI GATEWAY ARCHITECTURE
# ==================================================

print("UNIFIED AI GATEWAY ARCHITECTURE")
print()

print("                    Silverwing Agent")
print("                           ↓")
print("                      AI Gateway")
print("                           ↓")
print("                       Router")
print("                           ↓")
print("                    Capability Match")
print("                           ↓")
print("                    Provider Adapter")
print("                           ↓")
print("        ┌──────────────────┼──────────────────┐")
print("        ↓                  ↓                  ↓")
print("      Local              Cloud            Specialist")
print("        ↓                  ↓                  ↓")
print("      Model              Model              Model")
print("        └──────────────────┼──────────────────┘")
print("                           ↓")
print("                       AI Result")
print("                           ↓")
print("                         Agent")

print()


# ==================================================
# 23. PROVIDER INDEPENDENCE
# ==================================================

print("PROVIDER INDEPENDENCE")
print()

print(
    "The agent uses one gateway interface."
)

print()

print(
    "The gateway selects a provider."
)

print()

print(
    "The provider adapter translates the "
    "common request into provider-specific "
    "operations."
)

print()

print(
    "The gateway converts the provider result "
    "back into a common AIResponse."
)

print()


# ==================================================
# 24. FUTURE PROVIDER TYPES
# ==================================================

print("FUTURE PROVIDER TYPES")
print()

providers = [
    "local Transformers",
    "Ollama",
    "llama.cpp",
    "vLLM",
    "OpenAI-compatible APIs",
    "Anthropic-compatible APIs",
    "Hugging Face endpoints",
    "specialized vision models",
    "speech models",
    "embedding providers",
    "Silverwing ML services"
]


for provider in providers:

    print(
        "-",
        provider
    )

print()


# ==================================================
# 25. IMPORTANT DESIGN PRINCIPLE
# ==================================================

print("IMPORTANT DESIGN PRINCIPLE")
print()

print(
    "Do not let the agent become tightly coupled "
    "to provider-specific SDKs."
)

print()

print(
    "Provider-specific details belong inside "
    "provider adapters."
)

print()

print(
    "The gateway should expose a stable interface "
    "to the rest of Silverwing."
)

print()


# ==================================================
# 26. SILVERWING AI ARCHITECTURE
# ==================================================

print("SILVERWING AI ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Conversation Manager")
print(" ↓")
print("Agent / Planner")
print(" ↓")
print("AI Gateway")
print(" ↓")
print("Model Router")
print(" ↓")
print("Provider Adapter")
print(" ↓")
print("Selected AI Model")
print(" ↓")
print("AI Response")
print(" ↓")
print("Agent")
print(" ↓")
print("User")

print()


# ==================================================
# 27. CURRENT PROGRESS
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
print("Model Routing")
print(" ↓")
print("Provider Adapters")
print(" ↓")
print("Unified AI Gateway")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 58 COMPLETE ===")