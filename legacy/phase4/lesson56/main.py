# Silverwing ML
# Phase 4 - Lesson 56
# Service Discovery, Health Monitoring
# and Capability Detection


import json
import time
import uuid

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 56")
print("Service Discovery, Health Monitoring and Capability Detection")
print()


# ==================================================
# 1. UTILITIES
# ==================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. SERVICE RECORD
# ==================================================

class ServiceRecord:

    def __init__(
            self,
            name: str,
            base_url: str
    ):

        self.name = name

        self.base_url = (
            base_url.rstrip("/")
        )

        self.status = "unknown"

        self.capabilities = []

        self.model = {}

        self.last_checked = None

        self.latency_ms = None

        self.error = None

        self.check_count = 0


    def to_dict(self):

        return {
            "name":
                self.name,

            "base_url":
                self.base_url,

            "status":
                self.status,

            "capabilities":
                self.capabilities,

            "model":
                self.model,

            "last_checked":
                self.last_checked,

            "latency_ms":
                self.latency_ms,

            "error":
                self.error,

            "check_count":
                self.check_count
        }


# ==================================================
# 3. SERVICE MANAGER
# ==================================================

class ServiceManager:
    """
    Maintains a live registry of remote services.
    """

    def __init__(
            self,
            timeout=5.0
    ):

        self.timeout = timeout

        self.services = {}


    # ----------------------------------------------
    # Register
    # ----------------------------------------------

    def register(
            self,
            name: str,
            base_url: str
    ):

        self.services[name] = (
            ServiceRecord(
                name,
                base_url
            )
        )


    # ----------------------------------------------
    # Get service
    # ----------------------------------------------

    def get(
            self,
            name: str
    ):

        return self.services.get(
            name
        )


    # ----------------------------------------------
    # List services
    # ----------------------------------------------

    def list_services(self):

        return list(
            self.services.values()
        )


    # ----------------------------------------------
    # Health check
    # ----------------------------------------------

    def check_health(
            self,
            service: ServiceRecord
    ):

        start = time.perf_counter()

        service.check_count += 1

        service.last_checked = (
            utc_now()
        )

        service.error = None


        try:

            response = requests.get(
                (
                    f"{service.base_url}"
                    "/health"
                ),
                timeout=self.timeout
            )


            response.raise_for_status()


            data = response.json()


            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            service.latency_ms = (
                round(
                    duration,
                    2
                )
            )


            service.status = (
                data.get(
                    "status",
                    "unknown"
                )
            )


            return data


        except requests.Timeout as error:

            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            service.latency_ms = (
                round(
                    duration,
                    2
                )
            )


            service.status = (
                "timeout"
            )


            service.error = str(
                error
            )


            return {
                "status":
                    "timeout",

                "error":
                    str(error)
            }


        except requests.RequestException as error:

            duration = (
                               time.perf_counter()
                               -
                               start
                       ) * 1000


            service.latency_ms = (
                round(
                    duration,
                    2
                )
            )


            service.status = (
                "offline"
            )


            service.error = str(
                error
            )


            return {
                "status":
                    "offline",

                "error":
                    str(error)
            }


    # ----------------------------------------------
    # Capability discovery
    # ----------------------------------------------

    def discover_capabilities(
            self,
            service: ServiceRecord
    ):

        try:

            response = requests.get(
                (
                    f"{service.base_url}"
                    "/model"
                ),
                timeout=self.timeout
            )


            response.raise_for_status()


            data = response.json()


            service.model = data


            service.capabilities = [
                "machine_risk_prediction"
            ]


            return {
                "status":
                    "success",

                "capabilities":
                    service.capabilities,

                "model":
                    data
            }


        except requests.RequestException as error:

            service.capabilities = []

            service.model = {}

            service.error = str(
                error
            )


            return {
                "status":
                    "failed",

                "error":
                    str(error)
            }


    # ----------------------------------------------
    # Full service inspection
    # ----------------------------------------------

    def inspect(
            self,
            service: ServiceRecord
    ):

        health = self.check_health(
            service
        )


        capabilities = (
            self.discover_capabilities(
                service
            )
        )


        return {
            "service":
                service.name,

            "health":
                health,

            "capabilities":
                capabilities,

            "record":
                service.to_dict()
        }


    # ----------------------------------------------
    # Inspect all services
    # ----------------------------------------------

    def inspect_all(self):

        results = []


        for service in (
                self.list_services()
        ):

            results.append(
                self.inspect(
                    service
                )
            )


        return results


    # ----------------------------------------------
    # Capability search
    # ----------------------------------------------

    def find_capability(
            self,
            capability: str
    ):

        matches = []


        for service in (
                self.list_services()
        ):

            if capability in (
                    service.capabilities
            ):

                matches.append(
                    service
                )


        return matches


    # ----------------------------------------------
    # Available services
    # ----------------------------------------------

    def available_services(self):

        return [
            service
            for service
            in self.list_services()
            if service.status == "healthy"
        ]


    # ----------------------------------------------
    # Service map
    # ----------------------------------------------

    def capability_map(self):

        result = {}


        for service in (
                self.list_services()
        ):

            for capability in (
                    service.capabilities
            ):

                if capability not in result:

                    result[capability] = []


                result[capability].append(
                    service.name
                )


        return result


service_manager = ServiceManager(
    timeout=5
)


# ==================================================
# 4. CONFIGURE SERVICES
# ==================================================

print("TEST 1: Service Registration")
print()


# Real service from Lesson 29/55.
service_manager.register(
    "ml_service",
    "http://127.0.0.1:8000"
)


# We deliberately register two unavailable
# demonstration services so the system learns
# to distinguish healthy and unavailable services.

service_manager.register(
    "memory_service",
    "http://127.0.0.1:8010"
)


service_manager.register(
    "llm_service",
    "http://127.0.0.1:8020"
)


for service in (
        service_manager.list_services()
):

    print(
        service.name,
        "->",
        service.base_url
    )


print()


# ==================================================
# 5. INSPECT ALL SERVICES
# ==================================================

print("TEST 2: Service Inspection")
print()


inspection_results = (
    service_manager.inspect_all()
)


for result in inspection_results:

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print()


# ==================================================
# 6. HEALTH SUMMARY
# ==================================================

print("TEST 3: Health Summary")
print()


for service in (
        service_manager.list_services()
):

    print(
        service.name,
        "| status:",
        service.status,
        "| latency:",
        service.latency_ms,
        "ms"
    )


print()


# ==================================================
# 7. AVAILABLE SERVICES
# ==================================================

print("TEST 4: Available Services")
print()


available = (
    service_manager.available_services()
)


for service in available:

    print(
        "-",
        service.name
    )


print()


# ==================================================
# 8. CAPABILITY MAP
# ==================================================

print("TEST 5: Capability Map")
print()


capability_map = (
    service_manager.capability_map()
)


print(
    json.dumps(
        capability_map,
        indent=4
    )
)

print()


# ==================================================
# 9. FIND CAPABILITY
# ==================================================

print("TEST 6: Find Machine Prediction Capability")
print()


matching_services = (
    service_manager.find_capability(
        "machine_risk_prediction"
    )
)


for service in matching_services:

    print(
        service.name,
        "supports machine prediction."
    )


print()


# ==================================================
# 10. SERVICE SELECTION
# ==================================================

print("TEST 7: Capability-Based Service Selection")
print()


def select_service(
        manager: ServiceManager,
        capability: str
):

    candidates = (
        manager.find_capability(
            capability
        )
    )


    healthy_candidates = [
        service
        for service
        in candidates
        if service.status == "healthy"
    ]


    if not healthy_candidates:

        return None


    healthy_candidates.sort(
        key=lambda service:
        service.latency_ms
        if service.latency_ms is not None
        else float("inf")
    )


    return healthy_candidates[0]


selected_service = select_service(
    service_manager,
    "machine_risk_prediction"
)


if selected_service:

    print(
        "Selected:",
        selected_service.name
    )

    print(
        "Latency:",
        selected_service.latency_ms,
        "ms"
    )

else:

    print(
        "No healthy service available."
    )


print()


# ==================================================
# 11. SERVICE REQUEST THROUGH DISCOVERY
# ==================================================

print("TEST 8: Dynamic Service Selection")
print()


machine = {
    "temperature": 101,
    "pressure": 135,
    "rpm": 3100,
    "operating_hours": 4200
}


if selected_service:

    try:

        start = time.perf_counter()


        response = requests.post(
            (
                f"{selected_service.base_url}"
                "/predict"
            ),
            json=machine,
            timeout=5
        )


        response.raise_for_status()


        duration = (
                           time.perf_counter()
                           -
                           start
                   ) * 1000


        print(
            "Prediction:"
        )


        print(
            json.dumps(
                response.json(),
                indent=4
            )
        )


        print(
            "Request latency:",
            round(
                duration,
                2
            ),
            "ms"
        )


    except requests.RequestException as error:

        print(
            "Dynamic request failed:"
        )

        print(
            error
        )


else:

    print(
        "No service selected."
    )


print()


# ==================================================
# 12. SERVICE HEALTH SCORE
# ==================================================

print("TEST 9: Service Health Score")
print()


def calculate_health_score(
        service: ServiceRecord
):

    if service.status != "healthy":

        return 0.0


    score = 1.0


    if (
            service.latency_ms is not None
            and
            service.latency_ms > 500
    ):

        score -= 0.2


    if (
            service.latency_ms is not None
            and
            service.latency_ms > 1000
    ):

        score -= 0.2


    if service.error:

        score -= 0.5


    return max(
        0.0,
        min(
            1.0,
            score
        )
    )


for service in (
        service_manager.list_services()
):

    score = calculate_health_score(
        service
    )


    print(
        service.name,
        "->",
        score
    )


print()


# ==================================================
# 13. SERVICE RECORDS
# ==================================================

print("TEST 10: Live Service Records")
print()


for service in (
        service_manager.list_services()
):

    print(
        json.dumps(
            service.to_dict(),
            indent=4
        )
    )

    print()


# ==================================================
# 14. REFRESH MONITORING
# ==================================================

print("TEST 11: Health Refresh")
print()


print(
    "Refreshing service health..."
)


second_inspection = (
    service_manager.inspect_all()
)


for result in second_inspection:

    service_record = result[
        "record"
    ]


    print(
        service_record[
            "name"
        ],
        "->",
        service_record[
            "status"
        ]
    )


print()


# ==================================================
# 15. HEALTH CHANGE DETECTION
# ==================================================

print("TEST 12: Health Change Detection")
print()


def detect_health_changes(
        previous,
        current
):

    changes = []


    previous_map = {
        item["record"]["name"]:
            item["record"]
        for item
        in previous
    }


    current_map = {
        item["record"]["name"]:
            item["record"]
        for item
        in current
    }


    for name, current_record in (
            current_map.items()
    ):

        previous_record = (
            previous_map.get(
                name
            )
        )


        if previous_record is None:

            changes.append(
                {
                    "service":
                        name,

                    "change":
                        "new_service"
                }
            )


            continue


        if (
                previous_record["status"]
                !=
                current_record["status"]
        ):

            changes.append(
                {
                    "service":
                        name,

                    "change":
                        "status_changed",

                    "before":
                        previous_record[
                            "status"
                        ],

                    "after":
                        current_record[
                            "status"
                        ]
                }
            )


    return changes


health_changes = (
    detect_health_changes(
        inspection_results,
        second_inspection
    )
)


if health_changes:

    print(
        json.dumps(
            health_changes,
            indent=4
        )
    )

else:

    print(
        "No health-state changes detected."
    )


print()


# ==================================================
# 16. CAPABILITY CHANGE DETECTION
# ==================================================

print("TEST 13: Capability Change Detection")
print()


def compare_capabilities(
        previous,
        current
):

    changes = []


    previous_map = {
        item["record"]["name"]:
            set(
                item["record"][
                    "capabilities"
                ]
            )
        for item
        in previous
    }


    current_map = {
        item["record"]["name"]:
            set(
                item["record"][
                    "capabilities"
                ]
            )
        for item
        in current
    }


    all_services = (
            set(previous_map)
            |
            set(current_map)
    )


    for service_name in (
            all_services
    ):

        before = previous_map.get(
            service_name,
            set()
        )

        after = current_map.get(
            service_name,
            set()
        )


        added = (
                after
                -
                before
        )

        removed = (
                before
                -
                after
        )


        if added or removed:

            changes.append(
                {
                    "service":
                        service_name,

                    "added":
                        sorted(
                            added
                        ),

                    "removed":
                        sorted(
                            removed
                        )
                }
            )


    return changes


capability_changes = (
    compare_capabilities(
        inspection_results,
        second_inspection
    )
)


if capability_changes:

    print(
        json.dumps(
            capability_changes,
            indent=4
        )
    )

else:

    print(
        "No capability changes detected."
    )


print()


# ==================================================
# 17. SERVICE ROUTING DECISION
# ==================================================

print("TEST 14: Agent Routing Decision")
print()


routing_request = {
    "goal":
        "Predict machine risk",

    "required_capability":
        "machine_risk_prediction"
}


service = select_service(
    service_manager,
    routing_request[
        "required_capability"
    ]
)


routing_decision = {
    "request":
        routing_request,

    "selected_service":
        service.name
        if service
        else None,

    "service_url":
        service.base_url
        if service
        else None,

    "status":
        "ready"
        if service
        else "unavailable"
}


print(
    json.dumps(
        routing_decision,
        indent=4
    )
)

print()


# ==================================================
# 18. FAILOVER CONCEPT
# ==================================================

print("TEST 15: Failover Concept")
print()


print(
    "If the preferred service becomes unavailable:"
)

print()

print(
    "1. Detect unhealthy status."
)

print(
    "2. Remove it from active routing."
)

print(
    "3. Find another service with the capability."
)

print(
    "4. Route the request to the replacement."
)

print(
    "5. Record the routing decision."
)

print()


# ==================================================
# 19. LIVE CAPABILITY GRAPH
# ==================================================

print("LIVE CAPABILITY GRAPH")
print()

print("Service Registry")
print("       ↓")
print("Health Monitoring")
print("       ↓")
print("Capability Discovery")
print("       ↓")
print("Service Scoring")
print("       ↓")
print("Capability Map")
print("       ↓")
print("Agent Router")
print("       ↓")
print("Best Available Service")

print()


# ==================================================
# 20. FUTURE SERVICE REGISTRY
# ==================================================

print("FUTURE SILVERWING SERVICE REGISTRY")
print()

future_services = [
    "llm_gateway",
    "memory_service",
    "vector_service",
    "ml_service",
    "tool_service",
    "scheduler_service",
    "research_service",
    "vision_service",
    "speech_service",
    "computer_service"
]


for name in future_services:

    print(
        "-",
        name
    )

print()


# ==================================================
# 21. SELF-MONITORING ARCHITECTURE
# ==================================================

print("SELF-MONITORING SILVERWING")
print()

print("Silverwing")
print("    ↓")
print("Service Manager")
print("    ↓")
print("Health Checks")
print("    ↓")
print("Capability Discovery")
print("    ↓")
print("Performance Measurements")
print("    ↓")
print("Service Map")
print("    ↓")
print("Agent Router")

print()


# ==================================================
# 22. WHY THIS MATTERS
# ==================================================

print("WHY SERVICE DISCOVERY MATTERS")
print()

print(
    "The agent should not be hard-coded to assume "
    "that every capability is always available."
)

print()

print(
    "It should inspect the environment and determine "
    "which capabilities are currently usable."
)

print()

print(
    "That makes the architecture more modular, "
    "replaceable, and resilient."
)

print()


# ==================================================
# 23. PERSONAL AI CONNECTION
# ==================================================

print("PERSONAL AI CONNECTION")
print()

print(
    "A personal Silverwing installation may have "
    "different capabilities depending on which "
    "services, models, devices, or external APIs "
    "are currently available."
)

print()

print(
    "Service discovery gives the agent a live "
    "capability map instead of assuming a fixed "
    "environment."
)

print()


# ==================================================
# 24. CURRENT SILVERWING PROGRESS
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
print("Health Monitoring")
print(" ↓")
print("Capability-Aware Routing")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 56 COMPLETE ===")