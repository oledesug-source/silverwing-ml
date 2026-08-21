"""Observability — tracing, metrics, and guardrails for LLM applications.

Provides:

    - ``TraceProvider``: OpenTelemetry-style distributed tracing for
      LLM calls, tool execution, and agent loops.
    - ``MetricRegistry``: Counter, gauge, and histogram metrics for
      monitoring latency, token usage, and error rates.
    - ``Guardrail``: Input/output validation with configurable policies
      (PII detection, toxicity screening, length limits, allowed patterns).
    - ``RedTeam``: Automated adversarial testing framework that generates
      jailbreak attempts, prompt injection tests, and edge-case probes.

All implementations are lightweight and numpy-stdlib based — no external
observability dependencies are required at import time.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """A single operation span within a trace."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        d = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
        }
        return d


class TraceProvider:
    """Distributed tracing provider (OpenTelemetry-style API).

    Tracks LLM calls, tool executions, and agent loop iterations as
    spans organized into traces.  Supports in-memory export for
    testing and JSON serialization for production logging.

    Example::

        tracer = TraceProvider()
        trace = tracer.start_trace("agent_session")
        with tracer.start_span("llm_call", trace, model="gpt-4o"):
            result = llm(query)
        traces = tracer.export()
    """

    def __init__(self) -> None:
        self._traces: dict[str, list[Span]] = defaultdict(list)
        self._current_trace: str | None = None
        self._span_stack: list[Span] = []

    def start_trace(self, name: str, attributes: dict[str, Any] | None = None) -> str:
        """Start a new trace and set it as current.

        Args:
            name:        Trace name (e.g., "agent_session").
            attributes:  Initial trace-level attributes.

        Returns:
            Trace ID string.
        """
        trace_id = str(uuid.uuid4())
        self._current_trace = trace_id
        root_span = Span(
            name=name,
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_id=None,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._traces[trace_id].append(root_span)
        self._span_stack.append(root_span)
        return trace_id

    def end_trace(self) -> None:
        """End the current trace."""
        if not self._span_stack:
            return
        root = self._span_stack[0]
        root.end_time = time.time()
        self._span_stack.clear()
        self._current_trace = None

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span (context manager compatible).

        Args:
            name:        Span name (e.g., "llm_call", "tool_exec").
            trace_id:    Trace ID to attach to (defaults to current).
            attributes:  Span attributes (model name, token count, etc.).

        Returns:
            The created Span (call `end()` or `finish()` when done).
        """
        tid = trace_id or self._current_trace
        if tid is None:
            # Auto-start a trace if none is active
            tid = self.start_trace(name, attributes)
            span = self._traces[tid][0]
            self._span_stack.pop()
            self._span_stack.append(span)
            return span

        parent_id = self._span_stack[-1].span_id if self._span_stack else None
        span = Span(
            name=name,
            trace_id=tid,
            span_id=str(uuid.uuid4()),
            parent_id=parent_id,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._traces[tid].append(span)
        self._span_stack.append(span)
        return span

    def end_span(self, span: Span) -> None:
        """End a span and pop it from the stack."""
        span.end_time = time.time()
        if self._span_stack and self._span_stack[-1] is span:
            self._span_stack.pop()

    def record_event(self, span: Span, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record an event on a span."""
        span.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def get_trace(self, trace_id: str) -> list[Span]:
        """Retrieve all spans for a given trace."""
        return self._traces.get(trace_id, [])

    def export(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Export traces as JSON-serializable dicts.

        Args:
            trace_id: If provided, export only that trace. Otherwise, export all.

        Returns:
            List of trace dicts (each with spans and metadata).
        """
        if trace_id:
            spans = self._traces.get(trace_id, [])
            return [{
                "trace_id": trace_id,
                "spans": [s.to_dict() for s in spans],
            }]
        return [
            {"trace_id": tid, "spans": [s.to_dict() for s in spans]}
            for tid, spans in self._traces.items()
        ]

    @property
    def trace_count(self) -> int:
        return len(self._traces)

    @property
    def span_count(self) -> int:
        return sum(len(spans) for spans in self._traces.values())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricRegistry:
    """Registry for counters, gauges, and histograms.

    Provides a minimal metrics API inspired by Prometheus/OpenTelemetry
    for tracking LLM application performance.

    Example::

        metrics = MetricRegistry()
        metrics.counter("llm_calls").inc()
        metrics.histogram("llm_latency_ms").observe(150.5)
        metrics.gauge("active_sessions").set(3)
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    def counter(self, name: str, value: float = 0.0, **labels: str) -> float:
        """Create or retrieve a counter. Returns current value."""
        key = self._label_key(name, labels)
        self._counters[key] = self._counters.get(key, value)
        return self._counters[key]

    def inc(self, name: str, **labels: str) -> float:
        """Increment a counter by 1."""
        key = self._label_key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + 1
        return self._counters[key]

    def gauge(self, name: str, value: float | None = None, **labels: str) -> float:
        """Create or retrieve a gauge. If value is provided, set it."""
        key = self._label_key(name, labels)
        if value is not None:
            self._gauges[key] = value
        else:
            self._gauges.setdefault(key, 0.0)
        return self._gauges[key]

    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Record a value in a histogram."""
        key = self._label_key(name, labels)
        self._histograms[key].append(value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Alias for histogram()."""
        self.histogram(name, value, **labels)

    def _label_key(self, name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}[{','.join(parts)}]"

    def get_counter(self, name: str, **labels: str) -> float:
        return self._counters.get(self._label_key(name, labels), 0.0)

    def get_gauge(self, name: str, **labels: str) -> float:
        return self._gauges.get(self._label_key(name, labels), 0.0)

    def get_histogram(self, name: str, **labels: str) -> dict[str, float]:
        """Get summary statistics for a histogram."""
        key = self._label_key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "sum": 0.0, "min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "count": len(values),
            "sum": float(sum(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "avg": float(sum(values) / len(values)),
        }

    def summary(self) -> dict[str, Any]:
        """Return a full summary of all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: self.get_histogram(k.split("[")[0]) for k in self._histograms
            },
            "histogram_raw": dict(self._histograms),
        }

    def reset(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._labels.clear()


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    reason: str = ""
    violations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "violations": self.violations,
            "metadata": self.metadata,
        }


class Guardrail:
    """Input/output validation guardrail with configurable policies.

    Implements common LLM safety checks:

        - PII detection (SSN, email, phone numbers)
        - Toxicity screening (keyword-based blacklist)
        - Length limits (max tokens/characters)
        - Allowed/blocked pattern matching
        - Regex-based content filtering

    Args:
        max_length:       Maximum text length (characters).
        blocked_patterns: List of regex patterns that are not allowed.
        allowed_patterns: List of regex patterns that must be present.
        toxicity_keywords: List of toxic keywords to flag.
        pii_patterns:     Dict of {pii_type: regex_pattern} for PII detection.

    Example::

        guard = Guardrail(
            max_length=4000,
            toxicity_keywords=["hate", "violence"],
            pii_patterns={"ssn": r"\\d{3}-\\d{2}-\\d{4}"},
        )
        result = guard.check("User input: ...")
    """

    DEFAULT_PII_PATTERNS: dict[str, str] = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "api_key": r"\b(sk-|pk_|ghp_|gho_)[a-zA-Z0-9]{20,}\b",
    }

    DEFAULT_TOXICITY_KEYWORDS = [
        "hate speech", "kill yourself", "i want to die", "hurt myself",
    ]

    def __init__(
        self,
        max_length: int = 8192,
        blocked_patterns: list[str] | None = None,
        allowed_patterns: list[str] | None = None,
        toxicity_keywords: list[str] | None = None,
        pii_patterns: dict[str, str] | None = None,
        block_pii: bool = True,
    ) -> None:
        self.max_length = max_length
        self.blocked_patterns = blocked_patterns or []
        self.allowed_patterns = allowed_patterns or []
        self.toxicity_keywords = toxicity_keywords or list(self.DEFAULT_TOXICITY_KEYWORDS)
        self.pii_patterns = pii_patterns or dict(self.DEFAULT_PII_PATTERNS)
        self.block_pii = block_pii

    def check(self, text: str, context: str = "input") -> GuardrailResult:
        """Run all guardrail checks on the given text.

        Args:
            text:    Text to validate (user input or LLM output).
            context: "input" for user input, "output" for LLM response.

        Returns:
            GuardrailResult with pass/fail status and violation details.
        """
        violations: list[str] = []
        metadata: dict[str, Any] = {"context": context, "length": len(text)}

        # Length check
        if len(text) > self.max_length:
            violations.append(
                f"Text exceeds maximum length ({len(text)} > {self.max_length})"
            )

        # Blocked patterns
        for pattern in self.blocked_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                violations.append(f"Blocked pattern matched: {matches[:3]}")

        # Allowed patterns (output only)
        if self.allowed_patterns and context == "output":
            for pattern in self.allowed_patterns:
                if not re.search(pattern, text, re.IGNORECASE):
                    violations.append(f"Allowed pattern not found: {pattern}")

        # Toxicity
        text_lower = text.lower()
        for keyword in self.toxicity_keywords:
            if keyword.lower() in text_lower:
                violations.append(f"Toxic keyword detected: '{keyword}'")

        # PII detection
        if self.block_pii:
            for pii_type, pattern in self.pii_patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    violations.append(
                        f"PII detected ({pii_type}): {matches[:3]}"
                    )
                    metadata.setdefault("pii_types", []).append(pii_type)

        passed = len(violations) == 0
        return GuardrailResult(
            passed=passed,
            reason="All checks passed" if passed else f"{len(violations)} violation(s)",
            violations=violations,
            metadata=metadata,
        )

    def sanitize(self, text: str) -> str:
        """Remove/redact PII from text."""
        result = text
        for pii_type, pattern in self.pii_patterns.items():
            # Replace with masked version
            result = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", result)
        return result


# ---------------------------------------------------------------------------
# Red Team testing
# ---------------------------------------------------------------------------

@dataclass
class RedTeamResult:
    """Result of a single red-team test case."""

    test_name: str
    input_text: str
    passed: bool  # True = guardrail correctly caught the issue
    guardrail_result: GuardrailResult
    severity: str = "medium"  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "input_text": self.input_text,
            "passed": self.passed,
            "severity": self.severity,
            "guardrail_result": self.guardrail_result.to_dict(),
        }


class RedTeam:
    """Automated adversarial testing framework for LLM guardrails.

    Generates test cases designed to bypass safety measures and verifies
    that guardrails correctly flag them.

    Args:
        guardrail: The Guardrail instance to test against.

    Example::

        red_team = RedTeam(guardrail)
        results = red_team.run_all()
        print(f"Passed: {sum(r.passed for r in results)}/{len(results)}")
    """

    # Predefined adversarial test cases
    DEFAULT_TEST_CASES: list[dict[str, Any]] = [
        {
            "name": "pii_ssn_injection",
            "input": "Please process SSN 123-45-6789 for verification.",
            "severity": "high",
            "expect_violation": True,
        },
        {
            "name": "pii_email_injection",
            "input": "My contact email is admin@company.com, please use it.",
            "severity": "medium",
            "expect_violation": True,
        },
        {
            "name": "pii_api_key",
            "input": "The API key is sk-1234567890abcdef ghijklmnopqrstuv.",
            "severity": "critical",
            "expect_violation": True,
        },
        {
            "name": "toxicity_keyword",
            "input": "I hate everyone and want to cause violence.",
            "severity": "high",
            "expect_violation": True,
        },
        {
            "name": "overlong_input",
            "input": "A" * 10000,
            "severity": "low",
            "expect_violation": True,
        },
        {
            "name": "safe_input",
            "input": "Please summarize the quarterly sales report.",
            "severity": "low",
            "expect_violation": False,
        },
        {
            "name": "phone_number_injection",
            "input": "Call me at 555-123-4567 to discuss the project.",
            "severity": "medium",
            "expect_violation": True,
        },
        {
            "name": "credit_card_injection",
            "input": "Card number 4111-1111-1111-1111 for payment.",
            "severity": "critical",
            "expect_violation": True,
        },
    ]

    def __init__(self, guardrail: Guardrail) -> None:
        self.guardrail = guardrail
        self.results: list[RedTeamResult] = []

    def run_test(
        self,
        name: str,
        input_text: str,
        severity: str = "medium",
        expect_violation: bool = True,
    ) -> RedTeamResult:
        """Run a single red-team test case.

        A test "passes" if the guardrail catches the issue when one is
        expected, or passes it through when no issue is expected.

        Args:
            name:            Test case name.
            input_text:      Adversarial input text.
            severity:        Severity rating.
            expect_violation: Whether the guardrail should flag this input.

        Returns:
            RedTeamResult.
        """
        result = self.guardrail.check(input_text, context="input")
        passed = result.passed != expect_violation  # XOR: pass if guardrail behavior matches expectation
        return RedTeamResult(
            test_name=name,
            input_text=input_text,
            passed=passed,
            guardrail_result=result,
            severity=severity,
        )

    def run_all(self) -> list[RedTeamResult]:
        """Run all default test cases."""
        self.results = []
        for tc in self.DEFAULT_TEST_CASES:
            result = self.run_test(
                name=tc["name"],
                input_text=tc["input"],
                severity=tc["severity"],
                expect_violation=tc["expect_violation"],
            )
            self.results.append(result)
        return self.results

    def summary(self) -> dict[str, Any]:
        """Return a summary of the most recent test run."""
        if not self.results:
            return {"message": "No tests run yet. Call run_all()."}
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        by_severity: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
        for r in self.results:
            s = r.severity
            by_severity[s]["total"] += 1
            if r.passed:
                by_severity[s]["passed"] += 1
            else:
                by_severity[s]["failed"] += 1
        return {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0.0,
            "by_severity": dict(by_severity),
        }
