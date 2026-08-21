"""Main orchestrator — the heart of the Controlled Intelligence Platform.

``Orchestrator`` receives a user request, builds context, discovers
capabilities, invokes the model, and drives the bounded execution loop.
The LLM *proposes* — the orchestrator *decides* and *executes*.

Layer 4 integration: when ``policies`` (a ``silverwing_platform.policies.PolicyEngine``)
and ``approvals`` (a ``silverwing_platform.approvals.ApprovalManager``) are provided,
every capability action is gated by policy decisions before permission checks
and sandbox execution.  This enforces the invariant:

    LLM proposes → orchestrator coordinates → policy decides →
    permission authorizes → sandbox executes → audit records → user controls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from intelligence.tools.protocol import ToolCall, ToolResult
from sw_platform.audit.events import AuditEvent, AuditLog
from sw_platform.capabilities.registry import CapabilityRegistry
from sw_platform.context.builder import ContextBuilder
from sw_platform.context.models import RequestContext
from sw_platform.permissions.policy import PermissionEvaluator, PermissionPolicy
from sw_platform.sandbox.executor import SandboxExecutor

from .execution_loop import ExecutionLoop

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
    audit_events: list[AuditEvent] = field(default_factory=list)

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
            "audit_events": [e.to_dict() for e in self.audit_events],
        }


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class Orchestrator:
    """The Controlled Intelligence Platform orchestrator.

    Receives requests, builds context, discovers capabilities, and
    drives the bounded execution loop.  The LLM proposes — the
    orchestrator decides.

    Usage::

        orch = Orchestrator(registry=registry)
        response = orch.handle_request(ChatRequest(message="What is 2+2?"))
        print(response.text)
    """

    def __init__(
        self,
        registry: CapabilityRegistry | None = None,
        generator: Any = None,
        permissions: PermissionPolicy | None = None,
        sandbox: SandboxExecutor | None = None,
        audit: AuditLog | None = None,
        max_steps: int = 5,
        policies: Any = None,
        approvals: Any = None,
        database: Any = None,
    ) -> None:
        self._registry = registry or CapabilityRegistry()
        self._generator = generator
        self._evaluator = PermissionEvaluator(permissions or PermissionPolicy())
        self._sandbox = sandbox or SandboxExecutor()
        self._audit = audit or AuditLog()
        self._loop = ExecutionLoop(max_steps=max_steps)
        # Layer 4: policy engine and approval manager (lazy to avoid import errors)
        self._policies = policies
        self._approvals = approvals
        self._database = database
        self._auto_init_layer4()

    def _auto_init_layer4(self) -> None:
        """Auto-initialize Layer 4 components if not provided."""
        if self._policies is None:
            try:
                from silverwing_platform.policies import PolicyEngine
                self._policies = PolicyEngine(audit=self._audit)
            except Exception:
                self._policies = None
        if self._approvals is None:
            try:
                from silverwing_platform.approvals import ApprovalManager
                self._approvals = ApprovalManager(db=self._database)
            except Exception:
                self._approvals = None

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry

    @property
    def generator(self) -> Any:
        """The Layer 4 model provider (ModelProvider or legacy generator)."""
        return self._generator

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def policies(self) -> Any:
        return self._policies

    @property
    def approvals(self) -> Any:
        return self._approvals

    @property
    def database(self) -> Any:
        return self._database

    def handle_request(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request through the full orchestration loop."""
        t0 = time.monotonic()

        context = ContextBuilder.from_request(
            message=request.message,
            max_rounds=request.max_rounds,
            metadata=request.metadata,
        )

        # Persist session/conversation to database if available
        session_id = context.session.session_id
        conversation_id = None
        if self._database is not None:
            try:
                from silverwing_platform.database import (
                    ConversationRecord,
                    SessionRecord,
                )
                self._database.create_session(
                    SessionRecord(
                        session_id=session_id,
                        user_id=getattr(context.session, "user_id", "") or "",
                        project_id=request.metadata.get("project_id", ""),
                    )
                )
                conversation_id = self._database.create_conversation(
                    session_id=session_id,
                    title=request.message[:80],
                )
                self._database.add_message(conversation_id, "user", request.message)
            except Exception as exc:
                logger.warning("Database persistence failed: %s", exc)

        self._audit.record(AuditEvent(
            action="request_start",
            request_id=context.request_id,
            session_id=context.session.session_id,
            status="success",
            detail=request.message[:200],
        ))

        try:
            step_fn = self._make_step_fn(context)
            text, calls, results, rounds = self._loop.run(
                step_fn=step_fn,
                context=context,
                registry=self._registry,
                evaluator=self._evaluator,
                sandbox=self._sandbox,
                audit=self._audit,
                policies=self._policies,
                approvals=self._approvals,
            )
            elapsed = time.monotonic() - t0

            self._audit.record(AuditEvent(
                action="request_done",
                request_id=context.request_id,
                session_id=context.session.session_id,
                status="success",
                detail=f"rounds={rounds}",
                elapsed_ms=elapsed * 1000,
            ))

            # Persist assistant message to database
            if self._database is not None and conversation_id:
                try:
                    self._database.add_message(conversation_id, "assistant", text)
                except Exception:
                    pass

            events = self._audit.query(request_id=context.request_id)

            return ChatResponse(
                text=text,
                success=True,
                tool_calls=calls,
                tool_results=results,
                rounds=rounds,
                request_id=context.request_id,
                elapsed_seconds=elapsed,
                audit_events=events,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            self._audit.record(AuditEvent(
                action="request_error",
                request_id=context.request_id,
                session_id=context.session.session_id,
                status="error",
                detail=str(exc),
                elapsed_ms=elapsed * 1000,
            ))
            return ChatResponse(
                text="",
                success=False,
                error=str(exc),
                request_id=context.request_id,
                elapsed_seconds=elapsed,
            )

    def list_capabilities(self) -> list[dict[str, Any]]:
        """Return all registered capabilities as dicts."""
        return [
            {
                "name": cap.name,
                "version": cap.version,
                "description": cap.description,
                "input_schema": cap.input_schema,
                "risk_level": cap.risk_level,
                "timeout_seconds": cap.timeout_seconds,
                "enabled": cap.enabled,
                "tags": cap.tags,
                "source": cap.source,
                "capability_type": cap.capability_type,
            }
            for cap in self._registry.list(enabled_only=False)
        ]

    def _make_step_fn(
        self, context: RequestContext,
    ) -> Callable[[RequestContext, str | None], str]:
        """Create the step function for the execution loop."""
        def step_fn(ctx: RequestContext, override_prompt: str | None) -> str:
            return self._generate(ctx, override_prompt)
        return step_fn

    def _generate(
        self,
        context: RequestContext,
        override_prompt: str | None = None,
    ) -> str:
        """Generate a model response.

        Uses ModelProvider (Layer 4) if available, falls back to
        raw generate() call, then to fallback response.
        """
        if self._generator is None:
            return self._fallback_response(context)

        prompt = override_prompt or context.working_memory.build_context()
        if not prompt.strip():
            prompt = context.user_message

        # If generator is a ModelProvider (Layer 4), use InferenceRequest
        try:
            from silverwing_platform.models import ModelProvider
            if isinstance(self._generator, ModelProvider):
                from silverwing_platform.models import InferenceRequest, GenerationConfig
                req = InferenceRequest(
                    prompt=prompt,
                    config=GenerationConfig(
                        max_new_tokens=256,
                        temperature=0.0,
                        top_k=0,
                        top_p=1.0,
                    ),
                )
                resp = self._generator.infer(req)
                return resp.text
        except Exception as exc:
            logger.warning("ModelProvider infer failed, falling back: %s", exc)

        result = self._generator.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
        )
        return result.text

    def _fallback_response(self, context: RequestContext) -> str:
        """Fallback without a model — parse tool calls from user message."""
        tool_calls = self._registry.parse_calls(context.user_message)
        if tool_calls:
            results = []
            for call in tool_calls:
                cap = self._registry.get(call.tool_name)
                if cap is None:
                    results.append(f"Tool {call.tool_name}: unknown capability")
                    continue
                allowed, reason = self._evaluator.is_allowed(cap)
                if not allowed:
                    results.append(f"Tool {call.tool_name}: {reason}")
                    continue
                if cap.fn is not None:
                    result = self._sandbox.execute(
                        cap.fn, cap_id=call.tool_name, **call.args_dict,
                    )
                    if result.success:
                        results.append(f"Tool {call.tool_name} output: {result.output}")
                    else:
                        results.append(f"Tool {call.tool_name} error: {result.error}")
            return "\n".join(results)

        return f"I received your message: {context.user_message}"
