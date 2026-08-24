"""L1 basic responder and L2 router pattern."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .backend import LlmBackend
from .levels import AgentLevel, AgentTrace


def run_responder(
    backend: LlmBackend,
    message: str,
    *,
    system: str = "You are Silverwing, a helpful assistant.",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AgentTrace:
    """L1: single-shot generation, no tools, no memory."""
    t0 = time.monotonic()
    trace = AgentTrace(level=AgentLevel.BASIC_RESPONDER)
    text = backend.generate(message, system=system, max_tokens=max_tokens,
                            temperature=temperature)
    trace.add("generate", f"{len(text)} chars")
    trace.final_text = text
    trace.elapsed_seconds = time.monotonic() - t0
    return trace


@dataclass
class RouteTarget:
    """A named handler the router can dispatch to."""

    name: str
    keywords: tuple[str, ...]
    handler: Callable[[str], str]
    priority: int = 0
    metadata: dict = field(default_factory=dict)


class IntentRouter:
    """L2: classify an incoming message and dispatch to the best handler.

    Scoring is transparent: keyword hits (weighted by position and count)
    plus an explicit ``priority`` tiebreaker. Unrouted messages fall through
    to a default handler so the router never fails open.
    """

    def __init__(self, default: Callable[[str], str] | None = None) -> None:
        self._targets: list[RouteTarget] = []
        self._default = default or (lambda message: f"no route for: {message[:200]}")

    def register(self, target: RouteTarget) -> None:
        self._targets.append(target)

    def register_many(self, targets: list[RouteTarget]) -> None:
        self._targets.extend(targets)

    @property
    def routes(self) -> list[str]:
        return [t.name for t in self._targets]

    def score(self, target: RouteTarget, message: str) -> float:
        lowered = message.lower()
        total = 0.0
        for keyword in target.keywords:
            for match in re.finditer(re.escape(keyword.lower()), lowered):
                positional = 1.0 + 0.1 * (1.0 - match.start() / max(len(lowered), 1))
                total += positional
        return total * (1.0 + target.priority)

    def route(self, message: str) -> tuple[RouteTarget | None, float]:
        best: RouteTarget | None = None
        best_score = 0.0
        for target in self._targets:
            s = self.score(target, message)
            if s > best_score:
                best, best_score = target, s
        return best, best_score

    def handle(self, message: str) -> AgentTrace:
        t0 = time.monotonic()
        trace = AgentTrace(level=AgentLevel.ROUTER)
        target, confidence = self.route(message)
        trace.add("route", target.name if target else "<default>",
                  confidence=round(confidence, 3),
                  candidates=[t.name for t in self._targets])
        if target is None:
            trace.final_text = self._default(message)
        else:
            trace.final_text = target.handler(message)
        trace.success = True
        trace.elapsed_seconds = time.monotonic() - t0
        return trace


class LlmRouter(IntentRouter):
    """Router whose fallback handler consults an LLM instead of erroring."""

    def __init__(self, backend: LlmBackend) -> None:
        super().__init__(default=None)
        self._backend = backend
        self.register(RouteTarget(
            name="fallback_llm",
            keywords=(),
            handler=lambda m: self._backend.generate(m, system="You are Silverwing."),
            priority=-100,
        ))
