"""Bounded execution loop.

``ExecutionLoop`` drives the propose-validate-execute cycle with a hard
step limit.  Each iteration: generate → parse tool calls → permission
check → sandbox execute → feed results back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from intelligence.tools.protocol import ToolCall, ToolResult
from sw_platform.audit.events import AuditEvent, AuditLog
from sw_platform.capabilities.registry import CapabilityRegistry
from sw_platform.context.models import RequestContext
from sw_platform.permissions.policy import PermissionEvaluator
from sw_platform.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)


class ExecutionLoop:
    """Bounded propose-validate-execute loop.

    Usage::

        loop = ExecutionLoop(max_steps=5)
        text, calls, results, rounds = loop.run(
            step_fn=my_generate,
            context=request_context,
            registry=registry,
            evaluator=evaluator,
            sandbox=sandbox,
        )
    """

    def __init__(self, max_steps: int = 5) -> None:
        self._max_steps = max_steps

    @property
    def max_steps(self) -> int:
        return self._max_steps
    def run(
        self,
        step_fn: Callable[[RequestContext, str | None], str],
        context: RequestContext,
        registry: CapabilityRegistry,
        evaluator: PermissionEvaluator,
        sandbox: SandboxExecutor,
        audit: AuditLog | None = None,
        initial_output: str | None = None,
        policies: Any = None,
        approvals: Any = None,
    ) -> tuple[str, list[ToolCall], list[ToolResult], int]:
        """Run the bounded execution loop.

        Args:
             step_fn:          ``(context, override_prompt) -> model_output``
             context:          The request context.
             registry:         Capability registry for parsing/executing.
             evaluator:        Permission evaluator.
             sandbox:          Sandbox executor.
             audit:            Optional audit log.
             initial_output:   Pre-generated model output to start from.
             policies:         Optional :class:`silverwing_platform.policies.PolicyEngine`.
             approvals:        Optional :class:`silverwing_platform.approvals.ApprovalManager`.

        Returns:
            ``(final_text, all_tool_calls, all_tool_results, rounds_used)``
        """
        all_calls: list[ToolCall] = []
        all_results: list[ToolResult] = []
        model_output = initial_output if initial_output is not None else step_fn(context, None)

        for step in range(self._max_steps):
            tool_calls = registry.parse_calls(model_output)

            if not tool_calls:
                context.add_assistant_message(model_output)
                return model_output, all_calls, all_results, step

            round_results: list[ToolResult] = []
            for call in tool_calls:
                cap = registry.get(call.tool_name)
                if cap is None:
                    result = ToolResult(
                        tool_name=call.tool_name, output="",
                        success=False, error=f"Unknown capability: {call.tool_name}",
                    )
                else:
                    # --- Policy check (Phase 10) ---
                    if policies is not None:
                        from silverwing_platform.policies import PolicyDecision
                        decision, policy_reason = policies.evaluate(cap, context)
                        if decision == PolicyDecision.DENY:
                            result = ToolResult(
                                tool_name=call.tool_name, output="",
                                success=False,
                                error=f"Denied by policy ({policy_reason})",
                            )
                            if audit:
                                audit.record(AuditEvent(
                                    action="policy_denied",
                                    capability_id=call.tool_name,
                                    request_id=context.request_id,
                                    status="denied",
                                    detail=policy_reason,
                                ))
                            round_results.append(result)
                            all_calls.append(call)
                            all_results.append(result)
                            context.add_tool_result(result)
                            continue
                        if decision == PolicyDecision.REQUIRE_APPROVAL and approvals is not None:
                            req = approvals.request(
                                capability_id=call.tool_name,
                                action="execute",
                                target=cap.name,
                                risk_level=cap.risk_level,
                                user_id=getattr(context.session, "user_id", "") or "",
                                session_id=context.session.session_id,
                                reason=f"{call.tool_name} requires approval ({policy_reason})",
                            )
                            result = ToolResult(
                                tool_name=call.tool_name, output="",
                                success=False,
                                error=(
                                    f"Action requires approval: request_id={req.request_id} "
                                    f"(status={req.status.value})"
                                ),
                            )
                            if audit:
                                audit.record(AuditEvent(
                                    action="approval_required",
                                    capability_id=call.tool_name,
                                    request_id=context.request_id,
                                    status="pending",
                                    detail=req.request_id,
                                ))
                            round_results.append(result)
                            all_calls.append(call)
                            all_results.append(result)
                            context.add_tool_result(result)
                            continue
                    # --- Permission check (Phase 9) ---
                    allowed, perm_reason = evaluator.is_allowed(cap)
                    if not allowed:
                        result = ToolResult(
                            tool_name=call.tool_name, output="",
                            success=False, error=perm_reason,
                        )
                        if audit:
                            audit.record(AuditEvent(
                                action="permission_denied",
                                capability_id=call.tool_name,
                                request_id=context.request_id,
                                status="denied",
                                detail=perm_reason,
                            ))
                    else:
                        # --- Sandbox execution (Phase 12) ---
                        if cap.fn is not None:
                            result = sandbox.execute(
                                cap.fn, cap_id=call.tool_name, **call.args_dict,
                            )
                        else:
                            result = registry.execute_call(call)

                round_results.append(result)
                all_calls.append(call)
                all_results.append(result)
                context.add_tool_result(result)

                if audit:
                    audit.record(AuditEvent(
                        action="tool_call",
                        capability_id=call.tool_name,
                        request_id=context.request_id,
                        status="success" if result.success else "error",
                        detail=result.output if result.success else result.error,
                    ))

            results_text = registry.format_results(round_results)
            combined = (
                model_output
                + "\n\n"
                + results_text
                + "\n\nBased on the tool results above, provide your final answer."
            )
            context.add_assistant_message(model_output)
            model_output = step_fn(context, combined)

        context.add_assistant_message(model_output)
        return model_output, all_calls, all_results, self._max_steps
