"""L3 tool calling, L4 multi-agent, L5 autonomous, L6 loop engineering."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .backend import LlmBackend
from .levels import AgentLevel, AgentTrace

TOOL_PROMPT_HEADER = """TOOLS:
{tools}

Respond with EXACTLY one line:
TOOL: <name> <json-arguments>
or
FINAL: <answer for the user>"""


def _tool_catalog(tools: list[dict[str, Any]]) -> str:
    lines = []
    for tool in tools:
        params = ",".join(tool.get("parameters", {}).keys())
        lines.append(f"- {tool['name']}({params}): {tool.get('description', '')}")
    return "\n".join(lines)


@runtime_checkable
class ToolRuntime(Protocol):
    """Execution surface the tool-calling level drives."""

    def catalog(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]: ...


class SwPlatformToolRuntime:
    """Adapter exposing sw_platform ToolProviders as a ToolRuntime.

    Every call flows through the platform's own providers, so sandboxing,
    permission levels and audit logging keep working unchanged.
    """

    def __init__(self, providers: list[Any]) -> None:
        self._providers = providers

    def catalog(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for provider in self._providers:
            for spec in provider.get_tools():
                tools.append({
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": dict(spec.parameters or {}),
                    "risk_level": getattr(spec, "risk_level", ""),
                    "permission_required": getattr(spec, "permission_required", ""),
                })
        return tools
    def call(self, name: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for provider in self._providers:
            if not any(t.name == name for t in provider.get_tools()):
                continue
            try:
                result = provider.execute(name, **kwargs)
            except Exception as exc:
                return [{
                    "tool": name,
                    "success": False,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }]
            results.append({
                "tool": name,
                "success": bool(result.success),
                "output": str(getattr(result, "output", ""))[:2000],
                "error": str(getattr(result, "error", "") or ""),
            })
            return results
        return [{"tool": name, "success": False, "output": "",
                 "error": f"unknown tool: {name}"}]


class ToolCallingAgent:
    """L3: ReAct-style loop — the model requests tools, the platform executes."""

    def __init__(self, backend: LlmBackend, runtime: ToolRuntime,
                 max_rounds: int = 5) -> None:
        self.backend = backend
        self.runtime = runtime
        self.max_rounds = max_rounds

    def _parse(self, text: str) -> tuple[str | None, dict[str, Any], str]:
        tool_match = re.search(r"TOOL:\s*(\S+)\s*(\{.*\})?", text, re.DOTALL)
        if tool_match:
            raw = tool_match.group(2) or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
            return tool_match.group(1), args, ""
        final_match = re.search(r"FINAL:\s*(.*)", text, re.DOTALL)
        if final_match:
            return None, {}, final_match.group(1).strip()
        return None, {}, text.strip()

    def run(self, message: str) -> AgentTrace:
        t0 = time.monotonic()
        trace = AgentTrace(level=AgentLevel.TOOL_CALLING)
        catalog = self.runtime.catalog()
        set_tools = getattr(self.backend, "set_tools", None)
        if callable(set_tools):
            set_tools(catalog)
        catalog_text = _tool_catalog(catalog)
        history = f"USER: {message}\n"
        trace.final_text = ""
        for round_index in range(1, self.max_rounds + 1):
            prompt = TOOL_PROMPT_HEADER.format(tools=catalog_text) + "\n\n" + history
            response = self.backend.generate(prompt, system="You are Silverwing.")
            name, args, final = self._parse(response)
            if name is None:
                trace.add("final", (final or "")[:120], rounds=round_index)
                trace.final_text = final or "(empty response)"
                break
            results = self.runtime.call(name, args)
            observation = "; ".join(
                r["output"] if r["success"] else f"error: {r['error']}"
                for r in results
            )
            trace.add("tool_call", name, arguments=args,
                      results=results, rounds=round_index)
            history += f"ASSISTANT: {response.strip()}\nOBSERVATION: {observation}\n"
        else:
            trace.final_text = trace.final_text or "max tool rounds reached"
            trace.success = False
        trace.elapsed_seconds = time.monotonic() - t0
        return trace


@dataclass(frozen=True)
class AgentRole:
    """A named specialist with its own system prompt and routing keywords."""

    name: str
    system_prompt: str
    keywords: tuple[str, ...] = ()
    priority: int = 0


class MultiAgentOrchestrator:
    """L4: dispatch subtasks to specialised role-agents and merge output."""

    def __init__(self, backend: LlmBackend, roles: list[AgentRole]) -> None:
        if not roles:
            raise ValueError("at least one role required")
        self.backend = backend
        self.roles = roles

    def run(self, message: str) -> AgentTrace:
        t0 = time.monotonic()
        trace = AgentTrace(level=AgentLevel.MULTI_AGENT)
        lowered = message.lower()
        picked = [
            role for role in self.roles
            if any(kw.lower() in lowered for kw in role.keywords)
        ] or [max(self.roles, key=lambda r: r.priority)]
        contributions: dict[str, str] = {}
        for role in picked:
            reply = self.backend.generate(message, system=role.system_prompt)
            contributions[role.name] = reply
            trace.add("agent", role.name, contribution=reply[:200])
        trace.final_text = "\n".join(
            f"[{name}] {text}" for name, text in contributions.items()
        )
        trace.data = {"agents_engaged": list(contributions)}
        trace.elapsed_seconds = time.monotonic() - t0
        return trace


@dataclass
class PlanStep:
    index: int
    description: str
    status: str = "pending"
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status,
            "output": self.output[:300],
        }


class AutonomousGoalRunner:
    """L5: decompose a goal into steps, execute under budget + approval gates."""

    def __init__(
        self,
        backend: LlmBackend,
        step_runner: Callable[[str], str],
        *,
        max_steps: int = 8,
        budget_seconds: float = 300.0,
        approve: Callable[[PlanStep], bool] | None = None,
    ) -> None:
        self.backend = backend
        self.step_runner = step_runner
        self.max_steps = max_steps
        self.budget_seconds = budget_seconds
        self.approve = approve

    def _make_plan(self, goal: str) -> list[PlanStep]:
        prompt = (
            "PLAN: break the following goal into concise numbered steps.\n"
            f"GOAL: {goal}"
        )
        text = self.backend.generate(prompt, system="You are Silverwing planning.")
        steps: list[PlanStep] = []
        for line in text.splitlines():
            match = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
            if match:
                steps.append(PlanStep(index=len(steps) + 1,
                                      description=match.group(2).strip()))
            if len(steps) >= self.max_steps:
                break
        return steps

    def run(self, goal: str) -> AgentTrace:
        t0 = time.monotonic()
        trace = AgentTrace(level=AgentLevel.AUTONOMOUS)
        plan = self._make_plan(goal)
        if not plan:
            trace.success = False
            trace.final_text = "planner produced no steps"
            return trace
        trace.add("plan", f"{len(plan)} steps",
                  steps=[s.description for s in plan])
        aborted = False
        done = 0
        for step in plan:
            if time.monotonic() - t0 > self.budget_seconds:
                step.status = "out_of_budget"
                trace.add("budget", f"step {step.index} skipped")
                break
            if self.approve is not None and not self.approve(step):
                step.status = "rejected"
                trace.add("approval", f"step {step.index} rejected")
                aborted = True
                break
            step.output = str(self.step_runner(step.description) or "")[:2000]
            step.status = "done"
            done += 1
            trace.add("step", step.description, output=step.output[:200])
        trace.data = {"plan": [s.to_dict() for s in plan]}
        trace.final_text = (
            f"goal '{goal[:80]}': {done}/{len(plan)} steps completed"
            + (" (aborted at approval gate)" if aborted else "")
        )
        trace.success = not aborted and done == len(plan)
        trace.elapsed_seconds = time.monotonic() - t0
        return trace


class LoopEngineer:
    """L6: OODA outer loop — act, reflect, repair, repeat until converged."""

    def __init__(
        self,
        backend: LlmBackend,
        inner_runner: Callable[[str], str],
        *,
        max_cycles: int = 4,
    ) -> None:
        self.backend = backend
        self.inner_runner = inner_runner
        self.max_cycles = max_cycles
        self.memory: dict[str, str] = {}

    def _reflect(self, goal: str, last_output: str) -> tuple[bool, str]:
        prompt = (
            "REFLECT: evaluate whether the latest result satisfies the goal.\n"
            f"GOAL: {goal}\nRESULT: {last_output[:1500]}\n"
            "Answer with 'CRITIQUE:' then 'GOAL_MET: yes|no'."
        )
        text = self.backend.generate(prompt, system="You are Silverwing reflecting.")
        met = bool(re.search(r"GOAL_MET:\s*yes", text, re.IGNORECASE))
        critique_match = re.search(r"CRITIQUE:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        critique = (critique_match.group(1).strip() if critique_match else "")[:400]
        self.memory[f"reflection_{len(self.memory)}"] = critique
        return met, critique

    def run(self, goal: str) -> AgentTrace:
        t0 = time.monotonic()
        trace = AgentTrace(level=AgentLevel.LOOP_ENGINEERING)
        output = ""
        met = False
        cycles = 0
        for cycle in range(1, self.max_cycles + 1):
            cycles = cycle
            directive = goal if cycle == 1 else (
                f"{goal}\n(apply this feedback from the previous cycle: "
                f"{self.memory.get(f'reflection_{cycle - 2}', 'improve')})"
            )
            output = str(self.inner_runner(directive) or "")
            trace.add("cycle", f"cycle {cycle}", output=output[:200])
            met, critique = self._reflect(goal, output)
            trace.add("reflect", critique[:160], goal_met=met, cycle=cycle)
            if met:
                break
        trace.final_text = output
        trace.success = met
        trace.data = {"cycles": cycles, "memory": dict(self.memory),
                      "converged": met}
        trace.elapsed_seconds = time.monotonic() - t0
        return trace
