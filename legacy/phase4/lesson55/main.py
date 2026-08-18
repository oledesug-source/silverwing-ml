# Silverwing ML
# Phase 4 - Lesson 55
# Real Service-to-Service HTTP Communication
# Corrected Version


import json
import time
import uuid

from datetime import datetime, timezone
from typing import Any, Dict

import requests


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 55")
print("Real Service-to-Service HTTP Communication")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

ML_SERVICE_URL = "http://127.0.0.1:8000"

HEALTH_ENDPOINT = (
    f"{ML_SERVICE_URL}/health"
)

MODEL_ENDPOINT = (
    f"{ML_SERVICE_URL}/model"
)

PREDICT_ENDPOINT = (
    f"{ML_SERVICE_URL}/predict"
)


print("TEST 1: Configuration")
print()

print(
    "ML service:",
    ML_SERVICE_URL
)

print(
    "Health endpoint:",
    HEALTH_ENDPOINT
)

print(
    "Model endpoint:",
    MODEL_ENDPOINT
)

print(
    "Prediction endpoint:",
    PREDICT_ENDPOINT
)

print()


# ==================================================
# 2. MESSAGE CREATION
# ==================================================

def create_message(
        operation: str,
        payload: Dict[str, Any],
        sender: str = "silverwing_agent",
        receiver: str = "ml_service"
):

    return {
        "message_id": str(
            uuid.uuid4()
        ),
        "sender": sender,
        "receiver": receiver,
        "operation": operation,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "payload": payload
    }


# ==================================================
# 3. HTTP SERVICE CLIENT
# ==================================================

class MLServiceClient:
    """
    Direct client for the Silverwing ML API.
    """

    def __init__(
            self,
            base_url: str,
            timeout: float = 10.0
    ):

        self.base_url = base_url.rstrip(
            "/"
        )

        self.timeout = timeout


    def health(self):

        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()


    def model_info(self):

        response = requests.get(
            f"{self.base_url}/model",
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()


    def predict(
            self,
            temperature: float,
            pressure: float,
            rpm: float,
            operating_hours: float
    ):

        payload = {
            "temperature": temperature,
            "pressure": pressure,
            "rpm": rpm,
            "operating_hours":
                operating_hours
        }


        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()


    def get(
            self,
            endpoint: str
    ):

        response = requests.get(
            endpoint,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()


    def post(
            self,
            endpoint: str,
            payload: Dict[str, Any]
    ):

        response = requests.post(
            endpoint,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()


# ==================================================
# 4. CREATE CLIENT
# ==================================================

print("TEST 2: Create Service Client")
print()


client = MLServiceClient(
    ML_SERVICE_URL
)


print(
    "Client created."
)

print()


# ==================================================
# 5. HEALTH CHECK
# ==================================================

print("TEST 3: Service Health")
print()


try:

    health = client.health()


    print(
        json.dumps(
            health,
            indent=4
        )
    )


except requests.RequestException as error:

    print(
        "Health request failed:"
    )

    print(
        error
    )


print()


# ==================================================
# 6. MODEL INFORMATION
# ==================================================

print("TEST 4: Model Information")
print()


try:

    model_info = client.model_info()


    print(
        json.dumps(
            model_info,
            indent=4
        )
    )


except requests.RequestException as error:

    print(
        "Model request failed:"
    )

    print(
        error
    )


print()


# ==================================================
# 7. SINGLE PREDICTION
# ==================================================

print("TEST 5: Single Prediction")
print()


machine = {
    "temperature": 97,
    "pressure": 130,
    "rpm": 2600,
    "operating_hours": 3500
}


print(
    "Machine:"
)

print(
    json.dumps(
        machine,
        indent=4
    )
)

print()


try:

    prediction = client.predict(
        temperature=machine[
            "temperature"
        ],
        pressure=machine[
            "pressure"
        ],
        rpm=machine[
            "rpm"
        ],
        operating_hours=machine[
            "operating_hours"
        ]
    )


    print(
        "Prediction:"
    )

    print(
        json.dumps(
            prediction,
            indent=4
        )
    )


except requests.RequestException as error:

    print(
        "Prediction request failed:"
    )

    print(
        error
    )


print()


# ==================================================
# 8. STRUCTURED REQUEST
# ==================================================

print("TEST 6: Structured Service Request")
print()


message = create_message(
    operation="predict_machine_risk",
    payload=machine
)


print(
    json.dumps(
        message,
        indent=4
    )
)

print()


# ==================================================
# 9. HTTP HEADERS
# ==================================================

print("TEST 7: HTTP Headers")
print()


headers = {
    "Content-Type":
        "application/json",

    "X-Silverwing-Client":
        "silverwing-agent",

    "X-Request-ID":
        message["message_id"]
}


print(
    json.dumps(
        headers,
        indent=4
    )
)

print()


# ==================================================
# 10. RAW HTTP REQUEST
# ==================================================

print("TEST 8: Raw HTTP Request")
print()


try:

    response = requests.post(
        PREDICT_ENDPOINT,
        json=machine,
        headers=headers,
        timeout=10
    )


    print(
        "HTTP status:",
        response.status_code
    )


    selected_headers = {
        key: value
        for key, value
        in response.headers.items()
        if key.lower()
           in {
               "content-type",
               "content-length"
           }
    }


    print(
        "Response headers:"
    )

    print(
        json.dumps(
            selected_headers,
            indent=4
        )
    )

    print()

    response.raise_for_status()


    print(
        "Response body:"
    )

    print(
        response.text
    )


except requests.RequestException as error:

    print(
        "Raw HTTP request failed:"
    )

    print(
        error
    )


print()


# ==================================================
# 11. MULTIPLE MACHINE REQUESTS
# ==================================================

print("TEST 9: Multiple Service Requests")
print()


machines = [
    {
        "name":
            "Pump",

        "temperature":
            75,

        "pressure":
            110,

        "rpm":
            1500,

        "operating_hours":
            1500
    },

    {
        "name":
            "Compressor",

        "temperature":
            88,

        "pressure":
            125,

        "rpm":
            2400,

        "operating_hours":
            2800
    },

    {
        "name":
            "Generator",

        "temperature":
            105,

        "pressure":
            140,

        "rpm":
            3200,

        "operating_hours":
            4500
    }
]


results = []


for machine_data in machines:

    start = time.perf_counter()


    try:

        result = client.predict(
            temperature=machine_data[
                "temperature"
            ],
            pressure=machine_data[
                "pressure"
            ],
            rpm=machine_data[
                "rpm"
            ],
            operating_hours=machine_data[
                "operating_hours"
            ]
        )


        duration = (
                time.perf_counter()
                -
                start
        )


        results.append(
            {
                "name":
                    machine_data["name"],

                "prediction":
                    result.get(
                        "prediction"
                    ),

                "confidence":
                    result.get(
                        "confidence"
                    ),

                "duration_ms":
                    round(
                        duration * 1000,
                        2
                    )
            }
        )


    except requests.RequestException as error:

        results.append(
            {
                "name":
                    machine_data["name"],

                "error":
                    str(error)
            }
        )


for result in results:

    print(
        json.dumps(
            result,
            indent=4
        )
    )

print()


# ==================================================
# 12. SERVICE LATENCY
# ==================================================

print("TEST 10: Service Latency")
print()


latencies = [
    result["duration_ms"]
    for result in results
    if "duration_ms"
       in result
]


if latencies:

    average_latency = (
            sum(latencies)
            /
            len(latencies)
    )

    minimum_latency = min(
        latencies
    )

    maximum_latency = max(
        latencies
    )


    print(
        "Requests:",
        len(latencies)
    )

    print(
        "Average latency:",
        round(
            average_latency,
            2
        ),
        "ms"
    )

    print(
        "Minimum latency:",
        minimum_latency,
        "ms"
    )

    print(
        "Maximum latency:",
        maximum_latency,
        "ms"
    )

else:

    print(
        "No successful latency measurements."
    )


print()


# ==================================================
# 13. ERROR HANDLING
# ==================================================

print("TEST 11: Service Error Handling")
print()


def safe_predict(
        client,
        machine
):

    try:

        return {
            "status":
                "success",

            "data":
                client.predict(
                    temperature=machine[
                        "temperature"
                    ],
                    pressure=machine[
                        "pressure"
                    ],
                    rpm=machine[
                        "rpm"
                    ],
                    operating_hours=machine[
                        "operating_hours"
                    ]
                )
        }


    except requests.Timeout:

        return {
            "status":
                "timeout",

            "error":
                "ML service timed out."
        }


    except requests.ConnectionError:

        return {
            "status":
                "connection_error",

            "error":
                "Could not connect to ML service."
        }


    except requests.HTTPError as error:

        return {
            "status":
                "http_error",

            "error":
                str(error)
        }


    except requests.RequestException as error:

        return {
            "status":
                "request_error",

            "error":
                str(error)
        }


safe_result = safe_predict(
    client,
    machine
)


print(
    json.dumps(
        safe_result,
        indent=4
    )
)

print()


# ==================================================
# 14. SERVICE CAPABILITY OBJECT
# ==================================================

print("TEST 12: Service Capability Object")
print()


try:

    health = client.health()

    model = client.model_info()


    service_object = {
        "service":
            "ml_service",

        "endpoint":
            ML_SERVICE_URL,

        "status":
            health.get(
                "status"
            ),

        "model_loaded":
            health.get(
                "model_loaded"
            ),

        "capabilities": [
            "machine_risk_prediction"
        ],

        "model":
            model
    }


    print(
        json.dumps(
            service_object,
            indent=4
        )
    )


except requests.RequestException as error:

    print(
        "Could not construct service object:"
    )

    print(
        error
    )


print()


# ==================================================
# 15. REMOTE SERVICE ABSTRACTION
# ==================================================

print("TEST 13: Service Abstraction")
print()


class RemoteService:
    """
    Generic abstraction over the remote ML service.
    """

    def __init__(
            self,
            name: str,
            base_url: str
    ):

        self.name = name

        self.base_url = (
            base_url.rstrip("/")
        )


    # ----------------------------------------------
    # Health
    # ----------------------------------------------

    def health(self):

        response = requests.get(
            f"{self.base_url}/health",
            timeout=10
        )

        response.raise_for_status()

        return response.json()


    # ----------------------------------------------
    # Capabilities
    # ----------------------------------------------

    def capabilities(self):

        response = requests.get(
            f"{self.base_url}/model",
            timeout=10
        )

        response.raise_for_status()

        model = response.json()


        return {
            "service":
                self.name,

            "capabilities": [
                "predict_machine_risk"
            ],

            "model":
                model
        }


    # ----------------------------------------------
    # MACHINE PREDICTION
    # ----------------------------------------------

    def predict(
            self,
            temperature: float,
            pressure: float,
            rpm: float,
            operating_hours: float
    ):
        """
        This method was missing in the previous
        version and caused the AttributeError.
        """

        payload = {
            "temperature": temperature,
            "pressure": pressure,
            "rpm": rpm,
            "operating_hours":
                operating_hours
        }


        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        return response.json()


    # ----------------------------------------------
    # Generic request
    # ----------------------------------------------

    def request(
            self,
            method: str,
            endpoint: str,
            payload=None
    ):

        url = (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
        )


        if method.upper() == "GET":

            response = requests.get(
                url,
                timeout=10
            )

        elif method.upper() == "POST":

            response = requests.post(
                url,
                json=payload,
                timeout=10
            )

        else:

            raise ValueError(
                f"Unsupported HTTP method: {method}"
            )


        response.raise_for_status()

        return response.json()


ml_remote_service = RemoteService(
    "ml_service",
    ML_SERVICE_URL
)


try:

    print(
        ml_remote_service.health()
    )

    print(
        ml_remote_service.capabilities()
    )


except requests.RequestException as error:

    print(
        error
    )


print()


# ==================================================
# 16. REMOTE SERVICE REGISTRY
# ==================================================

print("TEST 14: Remote Service Registry")
print()


class RemoteServiceRegistry:

    def __init__(self):

        self.services = {}


    def register(
            self,
            service: RemoteService
    ):

        self.services[
            service.name
        ] = service


    def get(
            self,
            name
    ):

        return self.services.get(
            name
        )


    def list_services(self):

        return list(
            self.services.keys()
        )


remote_registry = (
    RemoteServiceRegistry()
)


remote_registry.register(
    ml_remote_service
)


print(
    "Registered remote services:"
)


for name in (
        remote_registry.list_services()
):

    print(
        "-",
        name
    )


print()


# ==================================================
# 17. AGENT TO REMOTE SERVICE
# ==================================================

print("TEST 15: Agent to Remote Service")
print()


agent_service = (
    remote_registry.get(
        "ml_service"
    )
)


if agent_service is None:

    print(
        "ERROR: ml_service was not found."
    )

else:

    try:

        result = agent_service.predict(
            temperature=101,
            pressure=135,
            rpm=3100,
            operating_hours=4200
        )


        print(
            "Agent received:"
        )

        print(
            json.dumps(
                result,
                indent=4
            )
        )


    except requests.RequestException as error:

        print(
            "Agent request failed:"
        )

        print(
            error
        )


print()


# ==================================================
# 18. REMOTE SERVICE HEALTH MAP
# ==================================================

print("TEST 16: Remote Service Health Map")
print()


health_map = {}


for service_name in (
        remote_registry.list_services()
):

    service = remote_registry.get(
        service_name
    )


    try:

        health_map[
            service_name
        ] = service.health()


    except requests.RequestException as error:

        health_map[
            service_name
        ] = {
            "status":
                "unavailable",

            "error":
                str(error)
        }


print(
    json.dumps(
        health_map,
        indent=4
    )
)

print()


# ==================================================
# 19. REMOTE SERVICE CAPABILITY MAP
# ==================================================

print("TEST 17: Capability Map")
print()


capability_map = {}


for service_name in (
        remote_registry.list_services()
):

    service = remote_registry.get(
        service_name
    )


    try:

        capability_map[
            service_name
        ] = service.capabilities()


    except requests.RequestException as error:

        capability_map[
            service_name
        ] = {
            "status":
                "unavailable",

            "error":
                str(error)
        }


print(
    json.dumps(
        capability_map,
        indent=4
    )
)

print()


# ==================================================
# 20. NETWORK ARCHITECTURE
# ==================================================

print("NETWORK SERVICE ARCHITECTURE")
print()

print("Silverwing Agent")
print("      ↓")
print("HTTP / JSON")
print("      ↓")
print("FastAPI ML Service")
print("      ↓")
print("Machine Learning Model")
print("      ↓")
print("JSON Prediction")
print("      ↓")
print("HTTP Response")
print("      ↓")
print("Silverwing Agent")

print()


# ==================================================
# 21. DISTRIBUTED SILVERWING
# ==================================================

print("DISTRIBUTED SILVERWING")
print()

print("                    Agent")
print("                      │")
print("             ┌────────┼────────┐")
print("             ↓        ↓        ↓")
print("           HTTP     HTTP      HTTP")
print("             ↓        ↓        ↓")
print("            ML      Memory     LLM")
print("          Service   Service   Service")
print("             │        │        │")
print("             ↓        ↓        ↓")
print("           Model    Database   Model")

print()


# ==================================================
# 22. SERVICE RESILIENCE
# ==================================================

print("SERVICE RESILIENCE")
print()

print(
    "Production service clients need:"
)

print(
    "- timeouts"
)

print(
    "- retries"
)

print(
    "- connection handling"
)

print(
    "- health checks"
)

print(
    "- structured errors"
)

print(
    "- request IDs"
)

print(
    "- observability"
)

print()


# ==================================================
# 23. FUTURE SERVICE DISCOVERY
# ==================================================

print("FUTURE SERVICE DISCOVERY")
print()

print("Agent")
print(" ↓")
print("Service Registry")
print(" ↓")
print("Health Checks")
print(" ↓")
print("Capability Discovery")
print(" ↓")
print("Available Service Map")
print(" ↓")
print("Agent Planner")

print()


# ==================================================
# 24. FUTURE SILVERWING SERVICE MESH
# ==================================================

print("FUTURE SILVERWING SERVICE MESH")
print()

print("Agent")
print(" │")
print(" ├── LLM Gateway")
print(" │")
print(" ├── Memory Service")
print(" │")
print(" ├── ML Service")
print(" │")
print(" ├── Tool Service")
print(" │")
print(" ├── Scheduler")
print(" │")
print(" ├── Research Service")
print(" │")
print(" └── Voice Service")

print()


# ==================================================
# 25. CURRENT PROGRESS
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
print("Message Protocol")
print(" ↓")
print("HTTP Service Communication")
print(" ↓")
print("Remote Service Abstraction")
print(" ↓")
print("Distributed AI Architecture")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 55 COMPLETE ===")