# Silverwing ML
# Phase 4 - Lesson 48
# Structured Tool Calling and Tool Schemas


import json
from datetime import datetime, timezone
from typing import Any, Callable


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 48")
print("Structured Tool Calling")
print()


# ==================================================
# 1. TOOL CLASS
# ==================================================

class StructuredTool:
    """
    Represents a callable AI tool with a
    machine-readable schema.
    """

    def __init__(
            self,
            name: str,
            description: str,
            parameters: dict,
            function: Callable[..., Any]
    ):

        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function


    def schema(self):

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


    def execute(
            self,
            arguments: dict
    ):

        return self.function(
            **arguments
        )


# ==================================================
# 2. TOOL IMPLEMENTATIONS
# ==================================================

def calculate(
        expression: str
):

    allowed_characters = (
        "0123456789+-*/(). "
    )

    if any(
            character
            not in allowed_characters
            for character in expression
    ):

        raise ValueError(
            "Unsupported characters in expression."
        )

    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

    except Exception as error:

        raise ValueError(
            f"Invalid expression: {error}"
        )

    if not isinstance(
            result,
            (int, float)
    ):

        raise ValueError(
            "Expression must produce a number."
        )

    return result


def analyze_machine(
        temperature: float,
        pressure: float,
        rpm: float
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
        "risk_score": score,
        "risk_level": level
    }


def get_time():

    return {
        "utc": datetime.now(
            timezone.utc
        ).isoformat()
    }


def summarize_text(
        text: str
):

    words = text.split()

    return {
        "character_count": len(text),
        "word_count": len(words),
        "preview": text[:120]
    }


# ==================================================
# 3. TOOL SCHEMAS
# ==================================================

calculator_schema = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": (
                "A basic arithmetic expression."
            )
        }
    },
    "required": [
        "expression"
    ],
    "additionalProperties": False
}


machine_schema = {
    "type": "object",
    "properties": {
        "temperature": {
            "type": "number",
            "description": (
                "Machine temperature."
            )
        },
        "pressure": {
            "type": "number",
            "description": (
                "Machine pressure."
            )
        },
        "rpm": {
            "type": "number",
            "description": (
                "Machine rotational speed."
            )
        }
    },
    "required": [
        "temperature",
        "pressure",
        "rpm"
    ],
    "additionalProperties": False
}


time_schema = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False
}


text_schema = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": (
                "Text to analyze."
            )
        }
    },
    "required": [
        "text"
    ],
    "additionalProperties": False
}


# ==================================================
# 4. CREATE STRUCTURED TOOLS
# ==================================================

print("TEST 1: Structured Tools")
print()


calculator_tool = StructuredTool(
    name="calculator",
    description=(
        "Perform basic arithmetic."
    ),
    parameters=calculator_schema,
    function=calculate
)


machine_tool = StructuredTool(
    name="machine_analyzer",
    description=(
        "Analyze machine operating conditions."
    ),
    parameters=machine_schema,
    function=analyze_machine
)


time_tool = StructuredTool(
    name="current_time",
    description=(
        "Return the current UTC time."
    ),
    parameters=time_schema,
    function=get_time
)


text_tool = StructuredTool(
    name="text_analyzer",
    description=(
        "Analyze basic properties of text."
    ),
    parameters=text_schema,
    function=summarize_text
)


print(
    "Created:"
)

print(
    calculator_tool.name
)

print(
    machine_tool.name
)

print(
    time_tool.name
)

print(
    text_tool.name
)

print()


# ==================================================
# 5. TOOL REGISTRY
# ==================================================

class StructuredToolRegistry:

    def __init__(self):

        self.tools = {}


    def register(
            self,
            tool: StructuredTool
    ):

        if tool.name in self.tools:

            raise ValueError(
                f"Tool already exists: {tool.name}"
            )

        self.tools[
            tool.name
        ] = tool


    def get(
            self,
            name: str
    ):

        return self.tools.get(
            name
        )


    def list_tools(self):

        return list(
            self.tools.values()
        )


    def schemas(self):

        return [
            tool.schema()
            for tool in self.list_tools()
        ]


registry = StructuredToolRegistry()

registry.register(
    calculator_tool
)

registry.register(
    machine_tool
)

registry.register(
    time_tool
)

registry.register(
    text_tool
)


# ==================================================
# 6. DISPLAY TOOL SCHEMAS
# ==================================================

print("TEST 2: Tool Schemas")
print()


print(
    json.dumps(
        registry.schemas(),
        indent=4
    )
)

print()


# ==================================================
# 7. TOOL CALL FORMAT
# ==================================================

print("TEST 3: Structured Tool Call")
print()


tool_call = {
    "tool_name": "machine_analyzer",
    "arguments": {
        "temperature": 97,
        "pressure": 130,
        "rpm": 2600
    }
}


print(
    json.dumps(
        tool_call,
        indent=4
    )
)

print()


# ==================================================
# 8. ARGUMENT VALIDATION
# ==================================================

def validate_arguments(
        schema,
        arguments
):

    if not isinstance(
            arguments,
            dict
    ):

        raise ValueError(
            "Arguments must be an object."
        )


    properties = schema.get(
        "properties",
        {}
    )


    required = schema.get(
        "required",
        []
    )


    for required_name in required:

        if required_name not in arguments:

            raise ValueError(
                f"Missing required argument: "
                f"{required_name}"
            )


    if (
            schema.get(
                "additionalProperties",
                True
            )
            is False
    ):

        unknown = (
                set(arguments)
                -
                set(properties)
        )

        if unknown:

            raise ValueError(
                "Unknown arguments: "
                +
                ", ".join(
                    sorted(unknown)
                )
            )


    for name, value in arguments.items():

        definition = properties.get(
            name
        )


        if definition is None:

            continue


        expected_type = definition.get(
            "type"
        )


        if expected_type == "string":

            if not isinstance(
                    value,
                    str
            ):

                raise ValueError(
                    f"{name} must be a string."
                )


        elif expected_type == "number":

            if not isinstance(
                    value,
                    (int, float)
            ):

                raise ValueError(
                    f"{name} must be a number."
                )


        elif expected_type == "integer":

            if not isinstance(
                    value,
                    int
            ):

                raise ValueError(
                    f"{name} must be an integer."
                )


        elif expected_type == "boolean":

            if not isinstance(
                    value,
                    bool
            ):

                raise ValueError(
                    f"{name} must be a boolean."
                )


    return True


# ==================================================
# 9. TOOL EXECUTOR
# ==================================================

class StructuredToolExecutor:

    def __init__(
            self,
            registry
    ):

        self.registry = registry


    def execute(
            self,
            call
    ):

        tool_name = call.get(
            "tool_name"
        )

        arguments = call.get(
            "arguments",
            {}
        )


        if not tool_name:

            raise ValueError(
                "tool_name is required."
            )


        tool = self.registry.get(
            tool_name
        )


        if tool is None:

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )


        validate_arguments(
            tool.parameters,
            arguments
        )


        try:

            result = tool.execute(
                arguments
            )

        except Exception as error:

            return {
                "status": "error",
                "tool": tool_name,
                "error": str(error)
            }


        return {
            "status": "success",
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        }


executor = StructuredToolExecutor(
    registry
)


# ==================================================
# 10. EXECUTE MACHINE TOOL
# ==================================================

print("TEST 4: Execute Structured Tool")
print()


result = executor.execute(
    tool_call
)


print(
    json.dumps(
        result,
        indent=4
    )
)

print()


# ==================================================
# 11. EXECUTE CALCULATOR
# ==================================================

print("TEST 5: Calculator Tool Call")
print()


calculator_call = {
    "tool_name": "calculator",
    "arguments": {
        "expression": "125 * 8 / 4"
    }
}


calculator_result = executor.execute(
    calculator_call
)


print(
    json.dumps(
        calculator_result,
        indent=4
    )
)

print()


# ==================================================
# 12. EXECUTE TIME TOOL
# ==================================================

print("TEST 6: Current Time Tool Call")
print()


time_call = {
    "tool_name": "current_time",
    "arguments": {}
}


time_result = executor.execute(
    time_call
)


print(
    json.dumps(
        time_result,
        indent=4
    )
)

print()


# ==================================================
# 13. EXECUTE TEXT TOOL
# ==================================================

print("TEST 7: Text Tool Call")
print()


text_call = {
    "tool_name": "text_analyzer",
    "arguments": {
        "text": (
            "Silverwing is an extensible "
            "personal AI system."
        )
    }
}


text_result = executor.execute(
    text_call
)


print(
    json.dumps(
        text_result,
        indent=4
    )
)

print()


# ==================================================
# 14. VALIDATION FAILURE
# ==================================================

print("TEST 8: Validation Failure")
print()


invalid_call = {
    "tool_name": "machine_analyzer",
    "arguments": {
        "temperature": "very hot",
        "pressure": 130,
        "rpm": 2600
    }
}


try:

    executor.execute(
        invalid_call
    )

except ValueError as error:

    print(
        "Validation error:"
    )

    print(
        error
    )


print()


# ==================================================
# 15. MISSING ARGUMENT
# ==================================================

print("TEST 9: Missing Argument")
print()


missing_argument_call = {
    "tool_name": "machine_analyzer",
    "arguments": {
        "temperature": 97,
        "pressure": 130
    }
}


try:

    executor.execute(
        missing_argument_call
    )

except ValueError as error:

    print(
        "Validation error:"
    )

    print(
        error
    )


print()


# ==================================================
# 16. UNKNOWN TOOL
# ==================================================

print("TEST 10: Unknown Tool")
print()


unknown_call = {
    "tool_name": "browser",
    "arguments": {
        "url": "https://example.com"
    }
}


try:

    executor.execute(
        unknown_call
    )

except ValueError as error:

    print(
        "Unknown tool error:"
    )

    print(
        error
    )


print()


# ==================================================
# 17. TOOL CALL RECORD
# ==================================================

print("TEST 11: Tool Call Record")
print()


tool_call_record = {
    "id": "call-001",
    "tool_name": "machine_analyzer",
    "arguments": {
        "temperature": 97,
        "pressure": 130,
        "rpm": 2600
    }
}


print(
    json.dumps(
        tool_call_record,
        indent=4
    )
)

print()


# ==================================================
# 18. TOOL RESULT RECORD
# ==================================================

print("TEST 12: Tool Result Record")
print()


tool_result_record = {
    "call_id": "call-001",
    "tool_name": "machine_analyzer",
    "status": "success",
    "result": {
        "risk_score": 30,
        "risk_level": "MEDIUM"
    }
}


print(
    json.dumps(
        tool_result_record,
        indent=4
    )
)

print()


# ==================================================
# 19. MULTI-STEP TOOL EXECUTION
# ==================================================

print("TEST 13: Multi-Step Tool Execution")
print()


calls = [

    {
        "tool_name": "calculator",
        "arguments": {
            "expression": "25 * 4"
        }
    },

    {
        "tool_name": "machine_analyzer",
        "arguments": {
            "temperature": 105,
            "pressure": 140,
            "rpm": 3200
        }
    },

    {
        "tool_name": "current_time",
        "arguments": {}
    }
]


for index, call in enumerate(
        calls,
        start=1
):

    result = executor.execute(
        call
    )


    print(
        "Call",
        index
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print()


# ==================================================
# 20. TOOL DISCOVERY
# ==================================================

print("TEST 14: Tool Discovery")
print()


def find_tools(
        keyword
):

    keyword = keyword.lower()


    matches = []


    for tool in registry.list_tools():

        searchable_text = (
                tool.name
                +
                " "
                +
                tool.description
        ).lower()


        if keyword in searchable_text:

            matches.append(
                tool
            )


    return matches


for query in [
    "machine",
    "text",
    "time",
    "arithmetic"
]:

    print(
        "Query:",
        query
    )


    matches = find_tools(
        query
    )


    for tool in matches:

        print(
            "-",
            tool.name
        )


    print()


# ==================================================
# 21. AGENT INTERACTION
# ==================================================

print("TEST 15: Agent Interaction")
print()


class StructuredAgent:

    def __init__(
            self,
            registry,
            executor
    ):

        self.registry = registry
        self.executor = executor


    def execute_tool_call(
            self,
            call
    ):

        return self.executor.execute(
            call
        )


agent = StructuredAgent(
    registry,
    executor
)


agent_call = {
    "tool_name": "machine_analyzer",
    "arguments": {
        "temperature": 101,
        "pressure": 135,
        "rpm": 3100
    }
}


agent_result = (
    agent.execute_tool_call(
        agent_call
    )
)


print(
    json.dumps(
        agent_result,
        indent=4
    )
)

print()


# ==================================================
# 22. LLM TOOL-CALLING FLOW
# ==================================================

print("LLM TOOL-CALLING FLOW")
print()

print("User request")
print("     ↓")
print("LLM reasoning")
print("     ↓")
print("Structured tool call")
print("     ↓")
print("Tool Registry")
print("     ↓")
print("Schema validation")
print("     ↓")
print("Permission check")
print("     ↓")
print("Tool execution")
print("     ↓")
print("Structured result")
print("     ↓")
print("LLM reasoning")
print("     ↓")
print("Final response")

print()


# ==================================================
# 23. TOOL SCHEMA CONCEPT
# ==================================================

print("TOOL SCHEMA")
print()

print(
    "A tool schema tells the model:"
)

print()

print(
    "1. What the tool is called."
)

print(
    "2. What it does."
)

print(
    "3. Which arguments it accepts."
)

print(
    "4. Which arguments are required."
)

print(
    "5. What types those arguments have."
)

print()


# ==================================================
# 24. WHY STRUCTURED CALLING MATTERS
# ==================================================

print("WHY STRUCTURED TOOL CALLING MATTERS")
print()

print(
    "Natural-language instructions are ambiguous."
)

print()

print(
    "Structured tool calls give software an "
    "explicit machine-readable action."
)

print()

print(
    "That makes validation, logging, permissions, "
    "retries, auditing, and execution much easier."
)

print()


# ==================================================
# 25. FUTURE SILVERWING TOOL CATEGORIES
# ==================================================

print("FUTURE SILVERWING TOOLS")
print()

future_tools = [
    "filesystem",
    "web_search",
    "browser",
    "database",
    "python",
    "machine_learning",
    "shell",
    "email",
    "messaging",
    "calendar",
    "vision",
    "speech",
    "code_repository",
    "deployment",
    "monitoring",
    "research",
    "computer_control"
]


for name in future_tools:

    print(
        "-",
        name
    )

print()


# ==================================================
# 26. ARCHITECTURAL PRINCIPLE
# ==================================================

print("ARCHITECTURAL PRINCIPLE")
print()

print(
    "The LLM should select capabilities."
)

print()

print(
    "The tool subsystem should execute capabilities."
)

print()

print(
    "The permission layer should control "
    "which capabilities may execute."
)

print()

print(
    "The result should return to the agent "
    "as structured information."
)

print()


# ==================================================
# 27. SILVERWING ARCHITECTURE
# ==================================================

print("SILVERWING TOOL ARCHITECTURE")
print()

print("                    LLM / AGENT")
print("                         │")
print("                  Tool Selection")
print("                         │")
print("                  Structured Call")
print("                         │")
print("                  Tool Registry")
print("                         │")
print("                 Schema Validation")
print("                         │")
print("                  Permission Layer")
print("                         │")
print("              ┌──────────┼──────────┐")
print("              ↓          ↓          ↓")
print("             ML        Memory     System")
print("            Tools       Tools      Tools")
print("              │          │          │")
print("              └──────────┼──────────┘")
print("                         ↓")
print("                  Structured Result")
print("                         ↓")
print("                       Agent")

print()


# ==================================================
# 28. CURRENT SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("LLM")
print(" ↓")
print("Conversation")
print(" ↓")
print("Persistent Memory")
print(" ↓")
print("Semantic Memory")
print(" ↓")
print("Vector Retrieval")
print(" ↓")
print("Tool Registry")
print(" ↓")
print("Structured Tool Calling")
print(" ↓")
print("Agent")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 48 COMPLETE ===")
