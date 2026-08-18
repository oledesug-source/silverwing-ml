# Silverwing ML
# Phase 4 - Lesson 59
# Real Local LLM Provider
#
# Goal:
# Connect a real Hugging Face causal language model
# to the Silverwing Unified AI Gateway.


import json
import time
import uuid

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 59")
print("Real Local LLM Provider")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "sshleifer/tiny-gpt2"

SEED = 42

set_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("TEST 1: Configuration")
print()

print(
    "Model:",
    MODEL_NAME
)

print(
    "Device:",
    DEVICE
)

print(
    "Seed:",
    SEED
)

print()


# ==================================================
# 2. LOAD TOKENIZER
# ==================================================

print("TEST 2: Load Tokenizer")
print()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    "Tokenizer:",
    type(tokenizer).__name__
)

print(
    "Vocabulary:",
    len(tokenizer)
)

print()


# ==================================================
# 3. LOAD LOCAL MODEL
# ==================================================

print("TEST 3: Load Local LLM")
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Model:",
    type(model).__name__
)

print(
    "Loaded on:",
    DEVICE
)

print()


# ==================================================
# 4. MODEL PARAMETERS
# ==================================================

print("TEST 4: Model Parameters")
print()


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    "Parameters:",
    parameter_count
)

print()


# ==================================================
# 5. AI REQUEST
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

    temperature: float = 0.8

    top_k: int = 50

    top_p: float = 0.95

    max_tokens: int = 40

    prefer_local: bool = True

    maximum_cost: Optional[float] = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 6. AI RESPONSE
# ==================================================

@dataclass
class AIResponse:

    provider: str

    model: str

    text: str

    request_id: str

    latency_ms: float

    usage: Dict[str, Any]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==================================================
# 7. PROVIDER INTERFACE
# ==================================================

class ProviderAdapter:

    @property
    def provider_name(self):

        raise NotImplementedError


    def list_models(self):

        raise NotImplementedError


    def health(self):

        raise NotImplementedError


    def generate(
            self,
            request,
            model_name
    ):

        raise NotImplementedError


# ==================================================
# 8. REAL LOCAL PROVIDER
# ==================================================

class LocalLLMProvider(
    ProviderAdapter
):
    """
    Real local Hugging Face provider.
    """

    def __init__(
            self,
            model,
            tokenizer,
            model_name,
            device
    ):

        self.model = model

        self.tokenizer = tokenizer

        self.model_name = model_name

        self.device = device


    @property
    def provider_name(self):

        return "local"


    # ----------------------------------------------
    # Model inventory
    # ----------------------------------------------

    def list_models(self):

        return [
            {
                "name":
                    self.model_name,

                "capabilities": [
                    "text_generation",
                    "completion"
                ],

                "local":
                    True,

                "cost":
                    0.0,

                "device":
                    str(
                        self.device
                    ),

                "parameters":
                    sum(
                        parameter.numel()
                        for parameter
                        in self.model.parameters()
                    )
            }
        ]


    # ----------------------------------------------
    # Health
    # ----------------------------------------------

    def health(self):

        return {
            "provider":
                self.provider_name,

            "status":
                "healthy",

            "model":
                self.model_name,

            "device":
                str(
                    self.device
                )
        }


    # ----------------------------------------------
    # Generation
    # ----------------------------------------------

    def generate(
            self,
            request: AIRequest,
            model_name: str
    ):

        if model_name != self.model_name:

            raise ValueError(
                "Requested model is not loaded "
                "by this provider."
            )


        start = time.perf_counter()


        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }


        input_token_count = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        with torch.no_grad():

            output_ids = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=(
                        request.max_tokens
                    ),

                    do_sample=True,

                    temperature=(
                        request.temperature
                    ),

                    top_k=(
                        request.top_k
                    ),

                    top_p=(
                        request.top_p
                    ),

                    pad_token_id=(
                        self.tokenizer.pad_token_id
                    )
                )
            )


        total_token_count = (
            output_ids.shape[1]
        )


        generated_token_count = (
                total_token_count
                -
                input_token_count
        )


        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )


        prompt_text = (
            request.prompt
        )


        if full_text.startswith(
                prompt_text
        ):

            response_text = (
                full_text[
                    len(prompt_text):
                ]
            ).strip()

        else:

            response_text = full_text


        duration = (
                           time.perf_counter()
                           -
                           start
                   ) * 1000


        return AIResponse(
            provider=self.provider_name,

            model=self.model_name,

            text=response_text,

            request_id=str(
                uuid.uuid4()
            ),

            latency_ms=round(
                duration,
                3
            ),

            usage={
                "input_tokens":
                    input_token_count,

                "generated_tokens":
                    generated_token_count,

                "total_tokens":
                    total_token_count
            },

            metadata={
                "device":
                    str(
                        self.device
                    ),

                "generation": {
                    "temperature":
                        request.temperature,

                    "top_k":
                        request.top_k,

                    "top_p":
                        request.top_p
                }
            }
        )


# ==================================================
# 9. CREATE REAL PROVIDER
# ==================================================

print("TEST 5: Create Real Provider")
print()


local_provider = LocalLLMProvider(
    model=model,
    tokenizer=tokenizer,
    model_name=MODEL_NAME,
    device=DEVICE
)


print(
    "Provider:",
    local_provider.provider_name
)

print()


# ==================================================
# 10. PROVIDER HEALTH
# ==================================================

print("TEST 6: Provider Health")
print()


print(
    json.dumps(
        local_provider.health(),
        indent=4
    )
)

print()


# ==================================================
# 11. MODEL INVENTORY
# ==================================================

print("TEST 7: Local Model Inventory")
print()


print(
    json.dumps(
        local_provider.list_models(),
        indent=4
    )
)

print()


# ==================================================
# 12. DIRECT MODEL GENERATION
# ==================================================

print("TEST 8: Direct Local Generation")
print()


direct_request = AIRequest(
    prompt=(
        "Silverwing is an artificial intelligence"
    ),
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    max_tokens=30
)


direct_response = local_provider.generate(
    direct_request,
    MODEL_NAME
)


print(
    "Generated text:"
)

print(
    direct_response.text
)

print()

print(
    "Latency:",
    direct_response.latency_ms,
    "ms"
)

print()


# ==================================================
# 13. PROVIDER REGISTRY
# ==================================================

class ProviderRegistry:

    def __init__(self):

        self.providers = {}


    def register(
            self,
            provider
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


provider_registry = (
    ProviderRegistry()
)


provider_registry.register(
    local_provider
)


# ==================================================
# 14. UNIFIED MODEL CATALOG
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

            for model_info in (
                    provider.list_models()
            ):

                item = dict(
                    model_info
                )


                item["provider"] = (
                    provider.provider_name
                )


                models.append(
                    item
                )


        return models


    def compatible_models(
            self,
            request
    ):

        compatible = []


        for model_info in (
                self.all_models()
        ):

            if not all(
                    capability
                    in model_info[
                        "capabilities"
                    ]
                    for capability
                    in request.required_capabilities
            ):

                continue


            if (
                    request.maximum_cost
                    is not None
                    and
                    model_info["cost"]
                    >
                    request.maximum_cost
            ):

                continue


            if (
                    request.prefer_local
                    and
                    not model_info["local"]
            ):

                continue


            compatible.append(
                model_info
            )


        return compatible


catalog = ModelCatalog(
    provider_registry
)


# ==================================================
# 15. ROUTER
# ==================================================

class ModelRouter:

    def __init__(
            self,
            catalog
    ):

        self.catalog = catalog


    def route(
            self,
            request
    ):

        candidates = (
            self.catalog.compatible_models(
                request
            )
        )


        if not candidates:

            return None


        # Local models are preferred when
        # the request asks for local execution.

        candidates.sort(
            key=lambda item: (
                item["local"],
                item["cost"] == 0,
                item["parameters"]
            ),
            reverse=True
        )


        return candidates[0]


router = ModelRouter(
    catalog
)


# ==================================================
# 16. UNIFIED AI GATEWAY
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
            request
    ):

        selected_model = (
            self.router.route(
                request
            )
        )


        if selected_model is None:

            raise RuntimeError(
                "No compatible model is available."
            )


        provider_name = (
            selected_model[
                "provider"
            ]
        )


        provider = self.registry.get(
            provider_name
        )


        if provider is None:

            raise RuntimeError(
                "Selected provider is unavailable."
            )


        response = provider.generate(
            request,
            selected_model["name"]
        )


        response.metadata[
            "gateway"
        ] = "silverwing"


        response.metadata[
            "selected_provider"
        ] = provider_name


        response.metadata[
            "selected_model"
        ] = selected_model[
            "name"
        ]


        return response


gateway = AIGateway(
    provider_registry,
    router
)


# ==================================================
# 17. GATEWAY GENERATION
# ==================================================

print("TEST 9: Unified Gateway Generation")
print()


gateway_request = AIRequest(
    prompt=(
        "Silverwing AI can help with"
    ),
    required_capabilities=[
        "text_generation"
    ],
    prefer_local=True,
    maximum_cost=0.0,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    max_tokens=25
)


gateway_response = gateway.generate(
    gateway_request
)


print(
    "Provider:",
    gateway_response.provider
)

print(
    "Model:",
    gateway_response.model
)

print(
    "Response:"
)

print(
    gateway_response.text
)

print()

print(
    "Request ID:",
    gateway_response.request_id
)

print(
    "Latency:",
    gateway_response.latency_ms,
    "ms"
)

print()


# ==================================================
# 18. MULTIPLE PROMPTS
# ==================================================

print("TEST 10: Multiple Gateway Requests")
print()


prompts = [
    "Machine learning is",
    "Artificial intelligence is",
    "Silverwing should",
    "A neural network learns"
]


for prompt in prompts:

    request = AIRequest(
        prompt=prompt,
        prefer_local=True,
        maximum_cost=0.0,
        max_tokens=20
    )


    response = gateway.generate(
        request
    )


    print(
        "Prompt:",
        prompt
    )

    print(
        "Response:",
        response.text
    )

    print(
        "Latency:",
        response.latency_ms,
        "ms"
    )

    print()


# ==================================================
# 19. ROUTING DECISION
# ==================================================

print("TEST 11: Routing Decision")
print()


routing_request = AIRequest(
    prompt=(
        "Explain transformers."
    ),
    required_capabilities=[
        "text_generation"
    ],
    prefer_local=True,
    maximum_cost=0.0
)


selected = router.route(
    routing_request
)


if selected:

    routing_decision = {
        "provider":
            selected["provider"],

        "model":
            selected["name"],

        "local":
            selected["local"],

        "cost":
            selected["cost"],

        "capabilities":
            selected["capabilities"]
    }


else:

    routing_decision = {
        "error":
            "No model selected."
    }


print(
    json.dumps(
        routing_decision,
        indent=4
    )
)

print()


# ==================================================
# 20. LOCAL MODEL STATUS
# ==================================================

print("TEST 12: Local Model Runtime")
print()


runtime_status = {
    "model":
        MODEL_NAME,

    "device":
        str(
            DEVICE
        ),

    "loaded":
        True,

    "training_mode":
        model.training,

    "evaluation_mode":
        not model.training
}


print(
    json.dumps(
        runtime_status,
        indent=4
    )
)

print()


# ==================================================
# 21. TOKENIZATION INSPECTION
# ==================================================

print("TEST 13: Tokenization Inspection")
print()


text = (
    "Silverwing is learning."
)


encoded = tokenizer(
    text,
    return_tensors="pt"
)


print(
    "Text:",
    text
)

print()

print(
    "Token IDs:"
)

print(
    encoded[
        "input_ids"
    ]
)

print()

print(
    "Token count:",
    encoded[
        "input_ids"
    ].shape[1]
)

print()


# ==================================================
# 22. MODEL INFORMATION
# ==================================================

print("TEST 14: Model Information")
print()


model_information = {
    "provider":
        local_provider.provider_name,

    "model":
        MODEL_NAME,

    "architecture":
        model.config.model_type,

    "parameters":
        parameter_count,

    "vocabulary":
        len(tokenizer),

    "device":
        str(
            DEVICE
        )
}


print(
    json.dumps(
        model_information,
        indent=4
    )
)

print()


# ==================================================
# 23. ERROR HANDLING
# ==================================================

print("TEST 15: Provider Error Handling")
print()


try:

    invalid_request = AIRequest(
        prompt="Test",
        required_capabilities=[
            "vision"
        ],
        prefer_local=True
    )


    gateway.generate(
        invalid_request
    )


except RuntimeError as error:

    print(
        "Expected routing error:"
    )

    print(
        error
    )


print()


# ==================================================
# 24. PROVIDER HEALTH MAP
# ==================================================

print("TEST 16: Provider Health Map")
print()


health_map = {}


for provider in (
        provider_registry.list_providers()
):

    health_map[
        provider.provider_name
    ] = provider.health()


print(
    json.dumps(
        health_map,
        indent=4
    )
)

print()


# ==================================================
# 25. UNIFIED AI ARCHITECTURE
# ==================================================

print("UNIFIED AI ARCHITECTURE")
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
print("REAL LOCAL LLM")
print(" ↓")
print("Generated Response")
print(" ↓")
print("Agent")
print(" ↓")
print("User")

print()


# ==================================================
# 26. CURRENT LOCAL AI PIPELINE
# ==================================================

print("CURRENT LOCAL AI PIPELINE")
print()

print("Prompt")
print(" ↓")
print("AIRequest")
print(" ↓")
print("Model Router")
print(" ↓")
print("Local Provider")
print(" ↓")
print("Tokenizer")
print(" ↓")
print("Tiny GPT-2")
print(" ↓")
print("Logits")
print(" ↓")
print("Sampling")
print(" ↓")
print("Generated Text")

print()


# ==================================================
# 27. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "The local model is intentionally tiny and "
    "is being used to verify the provider architecture."
)

print()

print(
    "Its language quality is not representative "
    "of modern production conversational models."
)

print()

print(
    "The important result of this lesson is that "
    "a real model is now connected to the same "
    "gateway abstraction used by future providers."
)

print()


# ==================================================
# 28. FUTURE LOCAL MODELS
# ==================================================

print("FUTURE LOCAL MODEL OPTIONS")
print()

future_models = [
    "small instruction-tuned models",
    "larger local LLMs",
    "code models",
    "vision-language models",
    "embedding models",
    "speech models",
    "specialized ML models"
]


for model_name in future_models:

    print(
        "-",
        model_name
    )

print()


# ==================================================
# 29. SILVERWING PROGRESS
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
print(" ↓")
print("REAL LOCAL LLM")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 59 COMPLETE ===")