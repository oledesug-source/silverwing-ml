# Silverwing ML
# Phase 4 - Lesson 47
# Tool Registry and Tool Calling


from datetime import datetime, timezone
from typing import Any, Callable


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 47")
print("Tool Registry and Tool Calling")
print()


# ==================================================
# 1. TOOL DEFINITIONS
# ==================================================

print("TEST 1: Tool Definition")
print()


class Tool:
    """
    Represents one capability available to Silverwing.
    """

    def __init__(
            self,
            name: str,
            description: str,
            function: Callable[..., Any]
    ):

        self.name = name
        self.description = description
        self.function = function


    def execute(
            self,
            **arguments
    ):

        return self.function(
            **arguments
        )


# ==================================================
# 2. EXAMPLE TOOLS
# ==================================================

print("TEST 2: Create Tools")
print()


def calculator(
        expression: str
):

    """
    Educational calculator.

    This lesson deliberately supports a small,
    controlled set of arithmetic operations instead
    of executing arbitrary Python code.
    """

    allowed_characters = (
        "0123456789+-*/(). "
    )


    if any(
            character
            not in allowed_characters
            for character in expression
    ):

        raise ValueError(
            "Expression contains unsupported characters."
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
            f"Invalid mathematical expression: {error}"
        )


    if not isinstance(
            result,
            (int, float)
    ):

        raise ValueError(
            "Expression did not produce a number."
        )


    return result


def get_current_time():

    return datetime.now(
        timezone.utc
    ).isoformat()


def analyze_machine(
        temperature: float,
        pressure: float,
        rpm: float
):

    risk_score = 0


    if temperature >= 100:

        risk_score += 40

    elif temperature >= 80:

        risk_score += 20


    if rpm > 3000:

        risk_score += 40

    elif rpm > 2500:

        risk_score += 15


    if pressure >= 160:

        risk_score += 20

    elif pressure >= 130:

        risk_score += 10


    if risk_score >= 70:

        risk_level = "CRITICAL"

    elif risk_score >= 40:

        risk_level = "HIGH"

    elif risk_score >= 20:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    return {
        "risk_score": risk_score,
        "risk_level": risk_level
    }


def summarize_text(
        text: str
):

    words = text.split()


    return {
        "characters": len(text),
        "words": len(words),
        "preview": text[:100]
    }


calculator_tool = Tool(
    name="calculator",
    description=(
        "Perform basic arithmetic calculations."
    ),
    function=calculator
)


time_tool = Tool(
    name="current_time",
    description=(
        "Return the current UTC timestamp."
    ),
    function=get_current_time
)


machine_tool = Tool(
    name="machine_analyzer",
    description=(
        "Analyze machine temperature, pressure, "
        "and RPM to estimate a risk category."
    ),
    function=analyze_machine
)


text_tool = Tool(
    name="text_summarizer",
    description=(
        "Return basic statistics and a preview "
        "for a text input."
    ),
    function=summarize_text
)


print(
    "Tools created:"
)

print(
    calculator_tool.name
)

print(
    time_tool.name
)

print(
    machine_tool.name
)

print(
    text_tool.name
)

print()


# ==================================================
# 3. TOOL REGISTRY
# ==================================================

print("TEST 3: Tool Registry")
print()


class ToolRegistry:
    """
    Central registry for all available tools.
    """

    def __init__(self):

        self._tools = {}


    def register(
            self,
            tool: Tool
    ):

        if tool.name in self._tools:

            raise ValueError(
                f"Tool already registered: {tool.name}"
            )


        self._tools[
            tool.name
        ] = tool


    def unregister(
            self,
            tool_name: str
    ):

        self._tools.pop(
            tool_name,
            None
        )


    def get(
            self,
            tool_name: str
    ):

        return self._tools.get(
            tool_name
        )


    def list_tools(self):

        return list(
            self._tools.values()
        )


    def schemas(self):

        return [
            {
                "name": tool.name,
                "description":
                    tool.description
            }
            for tool in self.list_tools()
        ]


registry = ToolRegistry()


registry.register(
    calculator_tool
)

registry.register(
    time_tool
)

registry.register(
    machine_tool
)

registry.register(
    text_tool
)


print(
    "Registered tools:"
)

for tool in registry.list_tools():

    print(
        "-",
        tool.name,
        ":",
        tool.description
    )

print()


# ==================================================
# 4. TOOL LOOKUP
# ==================================================

print("TEST 4: Tool Lookup")
print()


requested_tool = registry.get(
    "machine_analyzer"
)


if requested_tool:

    print(
        "Found:",
        requested_tool.name
    )

    print(
        "Description:",
        requested_tool.description
    )

else:

    print(
        "Tool not found."
    )


print()


# ==================================================
# 5. TOOL EXECUTION
# ==================================================

print("TEST 5: Tool Execution")
print()


calculator_result = registry.get(
    "calculator"
).execute(
    expression="25 * 4 + 10"
)


print(
    "Calculator result:",
    calculator_result
)

print()


time_result = registry.get(
    "current_time"
).execute()


print(
    "Current UTC time:",
    time_result
)

print()


machine_result = registry.get(
    "machine_analyzer"
).execute(
    temperature=97,
    pressure=130,
    rpm=2600
)


print(
    "Machine analysis:",
    machine_result
)

print()


text_result = registry.get(
    "text_summarizer"
).execute(
    text=(
        "Silverwing is a personal AI system "
        "with memory, tools and machine learning."
    )
)


print(
    "Text analysis:",
    text_result
)

print()


# ==================================================
# 6. TOOL CALL OBJECT
# ==================================================

print("TEST 6: Tool Call Representation")
print()


class ToolCall:
    """
    Represents a request to invoke a tool.
    """

    def __init__(
            self,
            tool_name: str,
            arguments: dict
    ):

        self.tool_name = tool_name
        self.arguments = arguments


calculator_call = ToolCall(
    "calculator",
    {
        "expression": "100 / 4"
    }
)


print(
    "Tool:",
    calculator_call.tool_name
)

print(
    "Arguments:",
    calculator_call.arguments
)

print()


# ==================================================
# 7. TOOL EXECUTOR
# ==================================================

print("TEST 7: Tool Executor")
print()


class ToolExecutor:
    """
    Executes validated tool calls through
    the registry.
    """

    def __init__(
            self,
            registry: ToolRegistry
    ):

        self.registry = registry


    def execute(
            self,
            tool_call: ToolCall
    ):

        tool = self.registry.get(
            tool_call.tool_name
        )


        if tool is None:

            raise ValueError(
                f"Unknown tool: "
                f"{tool_call.tool_name}"
            )


        result = tool.execute(
            **tool_call.arguments
        )


        return {
            "tool": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "result": result
        }


executor = ToolExecutor(
    registry
)


execution_result = executor.execute(
    calculator_call
)


print(
    execution_result
)

print()


# ==================================================
# 8. TOOL ERROR HANDLING
# ==================================================

print("TEST 8: Tool Error Handling")
print()


bad_call = ToolCall(
    "does_not_exist",
    {}
)


try:

    executor.execute(
        bad_call
    )

except ValueError as error:

    print(
        "Handled tool error:"
    )

    print(
        error
    )


print()


# ==================================================
# 9. TOOL SCHEMAS
# ==================================================

print("TEST 9: Tool Schemas")
print()


schemas = registry.schemas()


for schema in schemas:

    print(
        schema
    )

print()


# ==================================================
# 10. AI TOOL SELECTION CONCEPT
# ==================================================

print("TEST 10: Tool Selection")
print()


def select_tool(
        request: str
):

    text = request.lower()


    if any(
            word in text
            for word in [
                "calculate",
                "math",
                "multiply",
                "divide",
                "add"
            ]
    ):

        return "calculator"


    if (
            "time"
            in text
    ):

        return "current_time"


    if any(
            word in text
            for word in [
                "machine",
                "temperature",
                "pressure",
                "rpm",
                "risk"
            ]
    ):

        return "machine_analyzer"


    if any(
            word in text
            for word in [
                "text",
                "words",
                "summarize"
            ]
    ):

        return "text_summarizer"


    return None


requests_to_test = [
    "Calculate 25 times 8.",
    "What time is it?",
    "Analyze this machine.",
    "How many words are in this text?"
]


for request in requests_to_test:

    selected_tool = select_tool(
        request
    )


    print(
        "Request:",
        request
    )

    print(
        "Selected tool:",
        selected_tool
    )

    print()


# ==================================================
# 11. SIMULATED AGENT
# ==================================================

print("TEST 11: Simulated Agent")
print()


class SimpleAgent:
    """
    Very small demonstration of an agent loop.

    A real LLM agent would use model-generated
    tool calls instead of the simple rule-based
    selector used here.
    """

    def __init__(
            self,
            registry,
            executor
    ):

        self.registry = registry
        self.executor = executor


    def process(
            self,
            request,
            **arguments
    ):

        tool_name = select_tool(
            request
        )


        if tool_name is None:

            return {
                "status": "no_tool",
                "message": (
                    "No suitable tool found."
                )
            }


        call = ToolCall(
            tool_name,
            arguments
        )


        result = self.executor.execute(
            call
        )


        return {
            "status": "completed",
            "request": request,
            "tool_result": result
        }


agent = SimpleAgent(
    registry,
    executor
)


agent_result = agent.process(
    "Calculate something for me.",
    expression="12 * 8"
)


print(
    agent_result
)

print()


# ==================================================
# 12. TOOL PERMISSIONS
# ==================================================

print("TEST 12: Tool Permissions")
print()


class PermissionManager:
    """
    Simple capability policy.

    Production systems should have significantly
    more robust authorization and isolation.
    """

    def __init__(self):

        self.allowed_tools = {
            "calculator",
            "current_time",
            "machine_analyzer",
            "text_summarizer"
        }


    def can_execute(
            self,
            tool_name
    ):

        return (
                tool_name
                in self.allowed_tools
        )


permission_manager = (
    PermissionManager()
)


for tool in registry.list_tools():

    print(
        tool.name,
        "->",
        permission_manager.can_execute(
            tool.name
        )
    )

print()


# ==================================================
# 13. SECURE TOOL EXECUTOR
# ==================================================

print("TEST 13: Permission-Aware Execution")
print()


class SecureToolExecutor:

    def __init__(
            self,
            registry,
            permission_manager
    ):

        self.registry = registry

        self.permission_manager = (
            permission_manager
        )


    def execute(
            self,
            tool_call
    ):

        if not self.permission_manager.can_execute(
                tool_call.tool_name
        ):

            raise PermissionError(
                "Tool execution is not permitted: "
                + tool_call.tool_name
            )


        tool = self.registry.get(
            tool_call.tool_name
        )


        if tool is None:

            raise ValueError(
                "Unknown tool."
            )


        return tool.execute(
            **tool_call.arguments
        )


secure_executor = SecureToolExecutor(
    registry,
    permission_manager
)


secure_result = secure_executor.execute(
    ToolCall(
        "calculator",
        {
            "expression": "7 * 9"
        }
    )
)


print(
    "Secure result:",
    secure_result
)

print()


# ==================================================
# 14. TOOL RESULT STRUCTURE
# ==================================================

print("TEST 14: Structured Tool Result")
print()


structured_result = {
    "tool": "machine_analyzer",
    "status": "success",
    "timestamp": datetime.now(
        timezone.utc
    ).isoformat(),
    "result": machine_result
}


print(
    structured_result
)

print()


# ==================================================
# 15. AGENT LOOP
# ==================================================

print("AGENT LOOP")
print()

print("User request")
print("     ↓")
print("LLM / Agent reasoning")
print("     ↓")
print("Tool selection")
print("     ↓")
print("Permission check")
print("     ↓")
print("Tool execution")
print("     ↓")
print("Tool result")
print("     ↓")
print("LLM / Agent reasoning")
print("     ↓")
print("Final response")

print()


# ==================================================
# 16. SILVERWING TOOL ARCHITECTURE
# ==================================================

print("SILVERWING TOOL ARCHITECTURE")
print()

print("                 LLM / Agent")
print("                      │")
print("                Tool Registry")
print("                      │")
print("        ┌─────────────┼─────────────┐")
print("        ↓             ↓             ↓")
print("      ML Tool      Memory Tool    System Tool")
print("        ↓             ↓             ↓")
print("   Predictions    Retrieval      Operations")
print("        │             │             │")
print("        └─────────────┼─────────────┘")
print("                      ↓")
print("                 Tool Results")
print("                      ↓")
print("                 Agent / LLM")

print()


# ==================================================
# 17. FUTURE TOOL CATEGORIES
# ==================================================

print("FUTURE TOOL CATEGORIES")
print()

future_tools = [
    "filesystem",
    "web",
    "database",
    "code execution",
    "machine learning",
    "shell",
    "browser",
    "calendar",
    "email",
    "messaging",
    "computer control",
    "deployment",
    "monitoring",
    "research",
    "vision",
    "speech"
]


for tool_name in future_tools:

    print(
        "-",
        tool_name
    )

print()


# ==================================================
# 18. WHY TOOL REGISTRY MATTERS
# ==================================================

print("WHY TOOL REGISTRY MATTERS")
print()

print(
    "A registry gives the agent a common "
    "interface for discovering capabilities."
)

print()

print(
    "New capabilities can be added as tools "
    "without rewriting the entire AI core."
)

print()

print(
    "The same registry can later support "
    "plugins, connectors, services, and "
    "external APIs."
)

print()


# ==================================================
# 19. PERSONAL AI ARCHITECTURE
# ==================================================

print("PERSONAL AI ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Communicative Interface")
print(" ↓")
print("Agent / Reasoning Core")
print(" ↓")
print("Tool Registry")
print(" ↓")
print("Permissions")
print(" ↓")
print("Tools / Services")
print(" ↓")
print("Results")
print(" ↓")
print("Agent")
print(" ↓")
print("Response")

print()


# ==================================================
# 20. IMPORTANT DISTINCTION
# ==================================================

print("IMPORTANT DISTINCTION")
print()

print(
    "Tool calling is different from simply asking "
    "the LLM to describe an action."
)

print()

print(
    "A tool call produces a structured request "
    "that software can execute."
)

print()

print(
    "This is the foundation for real AI actions."
)

print()


# ==================================================
# 21. CURRENT SILVERWING PROGRESS
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
print("Agent")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 47 COMPLETE ===")