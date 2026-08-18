# Silverwing ML
# Phase 4 - Lesson 54
# Agent Communication Protocol


import json
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 54")
print("Agent Communication Protocol")
print()


# ==================================================
# 1. TIMESTAMP
# ==================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 2. MESSAGE TYPES
# ==================================================

MESSAGE_TYPES = {
    "request",
    "response",
    "event",
    "error",
    "heartbeat",
    "task"
}


print("TEST 1: Message Types")
print()

for message_type in MESSAGE_TYPES:

    print(
        "-",
        message_type
    )

print()


# ==================================================
# 3. MESSAGE STRUCTURE
# ==================================================

@dataclass
class AgentMessage:
    """
    Standard message passed between Silverwing
    components and services.
    """

    message_type: str

    sender: str

    receiver: str

    operation: str

    payload: Dict[str, Any]

    message_id: str = field(
        default_factory=lambda:
        str(uuid.uuid4())
    )

    correlation_id: Optional[str] = None

    timestamp: str = field(
        default_factory=utc_now
    )

    version: str = "1.0"


    def validate(self):

        if (
                self.message_type
                not in MESSAGE_TYPES
        ):

            raise ValueError(
                f"Unsupported message type: "
                f"{self.message_type}"
            )


        if not self.sender:

            raise ValueError(
                "Message sender is required."
            )


        if not self.receiver:

            raise ValueError(
                "Message receiver is required."
            )


        if not self.operation:

            raise ValueError(
                "Message operation is required."
            )


        if not isinstance(
                self.payload,
                dict
        ):

            raise ValueError(
                "Payload must be a dictionary."
            )


        return True


    def to_dict(self):

        self.validate()

        return {
            "version":
                self.version,

            "message_id":
                self.message_id,

            "correlation_id":
                self.correlation_id,

            "message_type":
                self.message_type,

            "sender":
                self.sender,

            "receiver":
                self.receiver,

            "operation":
                self.operation,

            "timestamp":
                self.timestamp,

            "payload":
                self.payload
        }


    def to_json(self):

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False
        )


    @classmethod
    def from_dict(
            cls,
            data
    ):

        message = cls(
            message_type=data[
                "message_type"
            ],

            sender=data[
                "sender"
            ],

            receiver=data[
                "receiver"
            ],

            operation=data[
                "operation"
            ],

            payload=data[
                "payload"
            ],

            message_id=data.get(
                "message_id",
                str(uuid.uuid4())
            ),

            correlation_id=data.get(
                "correlation_id"
            ),

            timestamp=data.get(
                "timestamp",
                utc_now()
            ),

            version=data.get(
                "version",
                "1.0"
            )
        )


        message.validate()

        return message


    @classmethod
    def from_json(
            cls,
            text
    ):

        data = json.loads(
            text
        )

        return cls.from_dict(
            data
        )


# ==================================================
# 4. CREATE MESSAGE
# ==================================================

print("TEST 2: Create Message")
print()


message = AgentMessage(
    message_type="request",
    sender="agent",
    receiver="ml_service",
    operation="predict_machine_risk",
    payload={
        "temperature": 97,
        "pressure": 130,
        "rpm": 2600
    }
)


print(
    json.dumps(
        message.to_dict(),
        indent=4
    )
)

print()


# ==================================================
# 5. SERIALIZATION
# ==================================================

print("TEST 3: Message Serialization")
print()


serialized = message.to_json()


print(
    serialized
)

print()


restored_message = (
    AgentMessage.from_json(
        serialized
    )
)


print(
    "Restored operation:",
    restored_message.operation
)

print(
    "Restored sender:",
    restored_message.sender
)

print()


# ==================================================
# 6. CORRELATION
# ==================================================

print("TEST 4: Request/Response Correlation")
print()


request = AgentMessage(
    message_type="request",
    sender="agent",
    receiver="ml_service",
    operation="predict_machine_risk",
    payload={
        "temperature": 105,
        "pressure": 140,
        "rpm": 3200
    }
)


response = AgentMessage(
    message_type="response",
    sender="ml_service",
    receiver="agent",
    operation="predict_machine_risk.result",
    payload={
        "risk_level": "CRITICAL",
        "risk_score": 80
    },
    correlation_id=request.message_id
)


print(
    "Request ID:",
    request.message_id
)

print()

print(
    "Response correlation ID:",
    response.correlation_id
)

print()


# ==================================================
# 7. MESSAGE VALIDATION
# ==================================================

print("TEST 5: Message Validation")
print()


valid = message.validate()


print(
    "Valid message:",
    valid
)

print()


# ==================================================
# 8. INVALID MESSAGE
# ==================================================

print("TEST 6: Invalid Message")
print()


try:

    invalid_message = AgentMessage(
        message_type="invalid_type",
        sender="agent",
        receiver="service",
        operation="test",
        payload={}
    )


    invalid_message.validate()


except ValueError as error:

    print(
        "Validation error:"
    )

    print(
        error
    )

print()


# ==================================================
# 9. SERVICE REGISTRY
# ==================================================

class ServiceRegistry:

    def __init__(self):

        self.services = {}


    def register(
            self,
            service_name,
            operations
    ):

        self.services[
            service_name
        ] = {
            "service":
                service_name,

            "operations":
                operations,

            "status":
                "online",

            "registered_at":
                utc_now()
        }


    def unregister(
            self,
            service_name
    ):

        self.services.pop(
            service_name,
            None
        )


    def get(
            self,
            service_name
    ):

        return self.services.get(
            service_name
        )


    def list_services(self):

        return list(
            self.services.values()
        )


registry = ServiceRegistry()


# ==================================================
# 10. REGISTER SERVICES
# ==================================================

print("TEST 7: Service Registry")
print()


registry.register(
    "agent",
    [
        "plan",
        "execute",
        "reason"
    ]
)


registry.register(
    "memory_service",
    [
        "store",
        "search",
        "retrieve"
    ]
)


registry.register(
    "ml_service",
    [
        "predict_machine_risk",
        "health"
    ]
)


registry.register(
    "tool_service",
    [
        "execute_tool",
        "list_tools"
    ]
)


for service in (
        registry.list_services()
):

    print(
        service
    )

print()


# ==================================================
# 11. SERVICE DISCOVERY
# ==================================================

print("TEST 8: Service Discovery")
print()


service = registry.get(
    "ml_service"
)


if service:

    print(
        "Found service:"
    )

    print(
        json.dumps(
            service,
            indent=4
        )
    )

else:

    print(
        "Service not found."
    )

print()


# ==================================================
# 12. SERVICE HEALTH
# ==================================================

class Service:

    def __init__(
            self,
            name,
            operations
    ):

        self.name = name

        self.operations = operations

        self.status = "online"


    def health(self):

        return {
            "service":
                self.name,

            "status":
                self.status,

            "timestamp":
                utc_now()
        }


ml_service = Service(
    "ml_service",
    [
        "predict_machine_risk",
        "health"
    ]
)


print("TEST 9: Service Health")
print()


print(
    ml_service.health()
)

print()


# ==================================================
# 13. MESSAGE ROUTER
# ==================================================

class MessageRouter:

    def __init__(
            self,
            registry
    ):

        self.registry = registry


    def route(
            self,
            message
    ):

        message.validate()


        service = self.registry.get(
            message.receiver
        )


        if service is None:

            return AgentMessage(
                message_type="error",
                sender="router",
                receiver=message.sender,
                operation="route_error",
                payload={
                    "error":
                        "Receiver service not found.",

                    "receiver":
                        message.receiver
                },
                correlation_id=(
                    message.message_id
                )
            )


        if (
                message.operation
                not in service["operations"]
        ):

            return AgentMessage(
                message_type="error",
                sender="router",
                receiver=message.sender,
                operation="operation_error",
                payload={
                    "error":
                        "Unsupported operation.",

                    "operation":
                        message.operation
                },
                correlation_id=(
                    message.message_id
                )
            )


        return None


router = MessageRouter(
    registry
)


# ==================================================
# 14. ROUTE VALID REQUEST
# ==================================================

print("TEST 10: Route Request")
print()


route_result = router.route(
    request
)


if route_result is None:

    print(
        "Request accepted by router."
    )

else:

    print(
        route_result.to_json()
    )

print()


# ==================================================
# 15. ROUTE INVALID SERVICE
# ==================================================

print("TEST 11: Unknown Service")
print()


unknown_service_request = AgentMessage(
    message_type="request",
    sender="agent",
    receiver="unknown_service",
    operation="test",
    payload={}
)


error_message = router.route(
    unknown_service_request
)


print(
    error_message.to_json()
)

print()


# ==================================================
# 16. ROUTE INVALID OPERATION
# ==================================================

print("TEST 12: Unsupported Operation")
print()


bad_operation_request = AgentMessage(
    message_type="request",
    sender="agent",
    receiver="ml_service",
    operation="delete_everything",
    payload={}
)


error_message = router.route(
    bad_operation_request
)


print(
    error_message.to_json()
)

print()


# ==================================================
# 17. SERVICE HANDLER
# ==================================================

class MLServiceHandler:

    def predict(
            self,
            temperature,
            pressure,
            rpm
    ):

        score = 0


        if temperature >= 100:

            score += 40

        elif temperature >= 80:

            score += 20


        if rpm > 3000:

            score += 40

        elif rpm > 2500:

            score += 15


        if pressure >= 160:

            score += 20

        elif pressure >= 130:

            score += 10


        if score >= 70:

            level = "CRITICAL"

        elif score >= 40:

            level = "HIGH"

        elif score >= 20:

            level = "MEDIUM"

        else:

            level = "LOW"


        return {
            "risk_score":
                score,

            "risk_level":
                level
        }


ml_handler = MLServiceHandler()


# ==================================================
# 18. MESSAGE PROCESSOR
# ==================================================

class MessageProcessor:

    def __init__(
            self,
            service_name,
            handlers
    ):

        self.service_name = (
            service_name
        )

        self.handlers = handlers


    def process(
            self,
            message
    ):

        try:

            message.validate()


            operation = (
                message.operation
            )


            handler = self.handlers.get(
                operation
            )


            if handler is None:

                return AgentMessage(
                    message_type="error",
                    sender=self.service_name,
                    receiver=message.sender,
                    operation=(
                            operation
                            +
                            ".error"
                    ),
                    payload={
                        "error":
                            "Operation handler not found."
                    },
                    correlation_id=(
                        message.message_id
                    )
                )


            result = handler(
                **message.payload
            )


            return AgentMessage(
                message_type="response",
                sender=self.service_name,
                receiver=message.sender,
                operation=(
                        operation
                        +
                        ".result"
                ),
                payload=result,
                correlation_id=(
                    message.message_id
                )
            )


        except Exception as error:

            return AgentMessage(
                message_type="error",
                sender=self.service_name,
                receiver=message.sender,
                operation="execution_error",
                payload={
                    "error":
                        str(error)
                },
                correlation_id=(
                    message.message_id
                )
            )


processor = MessageProcessor(
    service_name="ml_service",
    handlers={
        "predict_machine_risk":
            ml_handler.predict
    }
)


# ==================================================
# 19. COMPLETE REQUEST FLOW
# ==================================================

print("TEST 13: Complete Request/Response Flow")
print()


service_request = AgentMessage(
    message_type="request",
    sender="agent",
    receiver="ml_service",
    operation="predict_machine_risk",
    payload={
        "temperature": 105,
        "pressure": 140,
        "rpm": 3200
    }
)


routing_error = router.route(
    service_request
)


if routing_error is not None:

    final_message = routing_error

else:

    final_message = processor.process(
        service_request
    )


print(
    "REQUEST:"
)

print(
    json.dumps(
        service_request.to_dict(),
        indent=4
    )
)

print()

print(
    "RESPONSE:"
)

print(
    json.dumps(
        final_message.to_dict(),
        indent=4
    )
)

print()


# ==================================================
# 20. EVENT MESSAGE
# ==================================================

print("TEST 14: Event Message")
print()


event_message = AgentMessage(
    message_type="event",
    sender="ml_service",
    receiver="agent",
    operation="machine_risk_detected",
    payload={
        "machine_id":
            "machine-001",

        "risk_level":
            "CRITICAL"
    }
)


print(
    event_message.to_json()
)

print()


# ==================================================
# 21. HEARTBEAT MESSAGE
# ==================================================

print("TEST 15: Heartbeat")
print()


heartbeat = AgentMessage(
    message_type="heartbeat",
    sender="ml_service",
    receiver="agent",
    operation="heartbeat",
    payload={
        "status":
            "healthy"
    }
)


print(
    heartbeat.to_json()
)

print()


# ==================================================
# 22. COMMUNICATION PATTERN
# ==================================================

print("MESSAGE COMMUNICATION PATTERN")
print()

print("Agent")
print("  ↓")
print("Request Message")
print("  ↓")
print("Router")
print("  ↓")
print("Service")
print("  ↓")
print("Handler")
print("  ↓")
print("Result")
print("  ↓")
print("Response Message")
print("  ↓")
print("Agent")

print()


# ==================================================
# 23. WHY PROTOCOLS MATTER
# ==================================================

print("WHY COMMUNICATION PROTOCOLS MATTER")
print()

print(
    "A protocol defines how components exchange "
    "information without requiring them to share "
    "internal implementation details."
)

print()

print(
    "This makes services independently deployable "
    "and replaceable."
)

print()

print(
    "For example, the ML service can later be "
    "moved from local Python to FastAPI, another "
    "process, another machine, or a container "
    "without changing the conceptual message."
)

print()


# ==================================================
# 24. FUTURE TRANSPORTS
# ==================================================

print("FUTURE TRANSPORTS")
print()

transports = [
    "in-process messages",
    "HTTP",
    "REST",
    "WebSocket",
    "gRPC",
    "message queues",
    "event streams"
]


for transport in transports:

    print(
        "-",
        transport
    )

print()


# ==================================================
# 25. SERVICE MESH CONCEPT
# ==================================================

print("SERVICE COMMUNICATION")
print()

print("Agent")
print("  │")
print("  ├── Memory Service")
print("  │")
print("  ├── ML Service")
print("  │")
print("  ├── LLM Service")
print("  │")
print("  ├── Tool Service")
print("  │")
print("  └── Scheduler Service")

print()

print(
    "Each component communicates through "
    "defined interfaces."
)

print()


# ==================================================
# 26. FUTURE SILVERWING MESSAGE BUS
# ==================================================

print("FUTURE SILVERWING MESSAGE BUS")
print()

print("Component A")
print("     ↓")
print("Message Bus")
print("     ↓")
print("┌────┼────┬────┐")
print("↓    ↓    ↓    ↓")
print("ML  LLM Memory Tools")
print("└────┼────┴────┘")
print("     ↓")
print("Message Bus")
print("     ↓")
print("Component B")

print()


# ==================================================
# 27. IMPORTANT DESIGN PRINCIPLE
# ==================================================

print("DESIGN PRINCIPLE")
print()

print(
    "Keep reasoning, communication, execution, "
    "and storage loosely coupled."
)

print()

print(
    "A service should expose capabilities through "
    "a stable contract instead of exposing all "
    "of its internal implementation."
)

print()


# ==================================================
# 28. SILVERWING PROGRESS
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
print("Distributed AI Architecture")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 54 COMPLETE ===")