"""The core agent loop — the heart of the Intelligence Runtime.

``Orchestrator`` receives a user request, discovers capabilities,
invokes the model, parses tool calls, executes them in a sandbox,
feeds results back, and returns a final response.

The loop is deliberately simple and deterministic so that it can be
fully tested with a mock Generator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from intelligence.memory.context import WorkingMemory
from intelligence.tools.protocol import ToolCall, ToolResult

from .capabilities import CapabilityRegistry
from .context import RequestContext
from .permissions import PermissionCheck, PermissionPolicy
from .policies import PolicyEngine
from .sandbox import Sandbox

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Request / Response dataclasses
# ------------------------------------------------------------------

@dataclass
class ChatRequest:
    """Incoming chat request."""

    message: str
    max_rounds: int = 5
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Outgoing chat response."""

    text: str
    success: bool = True
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    rounds: int = 0
    request_id: str = ""
    elapsed_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "success": self.success,
            "tool_calls": [
                {"tool": tc.tool_name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "tool_results": [
                {
                    "tool": tr.tool_name,
                    "output": tr.output,
                    "success": tr.success,
                    "error": tr.error,
                }
                for tr in self.tool_results
            ],
            "rounds": self.rounds,
            "request_id": self.request_id,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "error": self.error,
        }


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class Orchestrator:
    """The agent loop — receives requests, discovers capabilities,
    invokes the planner/reasoner, calls tools, feeds results back,
    and returns a final response.

    Usage::

        agent = Agent.from_config()
        orch = Orchestrator(agent=agent)
        response = orch.handle_request(ChatRequest(message="What is 2+2?"))
        print(response.text)
    """

    def __init__(
        self,
        agent: Any = None,
        capability_registry: CapabilityRegistry | None = None,
        permissions: PermissionPolicy | None = None,
        policies: PolicyEngine | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        from .agents import Agent

        if agent is None:
            agent = Agent(capability_registry=capability_registry or CapabilityRegistry())
        self._agent = agent

        self._registry = capability_registry or agent.capability_registry
        self._permissions = PermissionCheck(permissions or PermissionPolicy())
        self._policies = policies or PolicyEngine()
        self._sandbox = sandbox or Sandbox()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_request(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request through the full orchestration loop.

        Flow:
            1. Build ``RequestContext``
            2. Store user message in working memory
            3. Generate model response
            4. Parse tool calls from response
            5. If tool calls exist: execute, feed back, repeat from 3
            6. Return final response
        """
        t0 = time.monotonic()

        context = RequestContext(
            user_message=request.message,
            working_memory=WorkingMemory(max_tokens=512),
            max_tool_rounds=request.max_rounds,
            metadata=request.metadata,
        )

        self._policies.audit(context, "request_start", request.message[:200])

        try:
            response = self._orchestration_loop(context)
        except Exception as exc:
            self._policies.audit(context, "request_error", str(exc))
            response = ChatResponse(
                text="",
                success=False,
                error=str(exc),
                request_id=context.request_id,
                elapsed_seconds=time.monotonic() - t0,
            )

        response.elapsed_seconds = time.monotonic() - t0
        response.request_id = context.request_id
        self._policies.audit(context, "request_done", f"rounds={response.rounds}")
        return response

    def list_capabilities(self) -> list[dict[str, Any]]:
        """Return all registered capabilities as dicts."""
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "parameters": cap.parameters,
                "source": cap.source,
                "tags": cap.tags,
            }
            for cap in self._registry.list_capabilities()
        ]

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _orchestration_loop(self, context: RequestContext) -> ChatResponse:
        """The core loop."""
        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        rounds = 0

        # Step 1: Generate initial response
        model_output = self._generate(context)

        while True:
            # Step 2: Parse tool calls
            tool_calls = self._registry.parse_calls(model_output)

            if not tool_calls:
                # No tool calls — we're done
                context.add_assistant_message(model_output)
                return ChatResponse(
                    text=model_output,
                    success=True,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    rounds=rounds,
                )

            # Step 3: Safety checks
            should_stop, reason = self._policies.should_stop(context, model_output)
            if should_stop:
                self._policies.audit(context, "loop_stop", reason)
                context.add_assistant_message(model_output)
                return ChatResponse(
                    text=model_output,
                    success=True,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    rounds=rounds,
                )

            # Step 4: Execute each tool call
            round_results: list[ToolResult] = []
            for call in tool_calls:
                # Permission check
                allowed, perm_reason = self._permissions.is_allowed(call.tool_name)
                if not allowed:
                    result = ToolResult(
                        tool_name=call.tool_name,
                        output="",
                        success=False,
                        error=perm_reason,
                    )
                    self._policies.audit(context, "permission_denied", perm_reason)
                else:
                    # Execute in sandbox
                    cap = self._registry.get(call.tool_name)
                    if cap is not None and cap.fn is not None:
                        result = self._sandbox.execute(
                            cap.fn,
                            tool_name=call.tool_name,
                            **call.args_dict,
                        )
                    else:
                        result = self._registry.execute_call(call)

                round_results.append(result)
                all_tool_calls.append(call)
                all_tool_results.append(result)
                context.add_tool_result(result)

            # Step 5: Format results and feed back
            results_text = self._registry.format_results(round_results)
            combined = (
                model_output
                + "\n\n"
                + results_text
                + "\n\nBased on the tool results above, provide your final answer."
            )

            # Store tool context and generate next response
            context.add_assistant_message(model_output)
            context.working_memory.add(
                __import__("intelligence.memory.context", fromlist=["MemoryEntry"]).MemoryEntry(
                    key=f"round-{rounds}",
                    content=results_text,
                    importance=0.85,
                )
            )

            rounds += 1
            model_output = self._generate(context, override_prompt=combined)

    def _generate(
        self,
        context: RequestContext,
        override_prompt: str | None = None,
    ) -> str:
        """Generate a model response."""
        if self._agent.generator is None:
            # No generator — synthesize a response by trying direct tool execution
            return self._fallback_response(context)

        prompt = override_prompt or context.working_memory.build_context()
        if not prompt.strip():
            prompt = context.user_message

        result = self._agent.generator.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
        )
        return result.text

    def _fallback_response(self, context: RequestContext) -> str:
        """Generate a response without a model — direct tool execution only."""
        # Try to detect and execute tool calls from the user message
        tool_calls = self._registry.parse_calls(context.user_message)
        if tool_calls:
            results = []
            for call in tool_calls:
                allowed, reason = self._permissions.is_allowed(call.tool_name)
                if not allowed:
                    results.append(f"Tool {call.tool_name}: {reason}")
                    continue
                cap = self._registry.get(call.tool_name)
                if cap and cap.fn:
                    result = self._sandbox.execute(
                        cap.fn, tool_name=call.tool_name, **call.args_dict
                    )
                    if result.success:
                        results.append(f"Tool {call.tool_name} output: {result.output}")
                    else:
                        results.append(f"Tool {call.tool_name} error: {result.error}")
            return "\n".join(results)

        # No tool calls detected — return a generic response
        return f"I received your message: {context.user_message}"
