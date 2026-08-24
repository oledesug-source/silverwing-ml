"""Pluggable LLM backends for the agentic stack.

Every level talks to an :class:`LlmBackend`, never to torch or HTTP directly,
so the same agent code runs against the locally served Silverwing endpoint,
a remote OpenAI-compatible server, or a deterministic stub in tests.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class LlmBackend(Protocol):
    """Minimal chat interface every backend must satisfy."""

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...


class ScriptedBackend:
    """Returns queued responses in order; extra calls repeat the last one."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        self.calls.append(prompt)
        if not self._responses:
            return ""
        if len(self.calls) <= len(self._responses):
            return self._responses[len(self.calls) - 1]
        return self._responses[-1]


class DeterministicBackend:
    """Offline fallback that answers deterministically without a model.

    Understands the agentic stack's own control syntax so levels remain
    exercisable without weights: prompts containing ``TOOLS:`` get a tool
    invocation, ``PLAN:`` requests get a numbered plan, everything else gets
    an echo-style reply.
    """

    def __init__(self, tools: list[dict[str, object]] | None = None) -> None:
        self._tool_names = [str(t["name"]) for t in (tools or [])]
        self._tool_rounds = 0

    def set_tools(self, tools: list[dict[str, object]]) -> None:
        """Refresh known tools (called by the tool-calling loop each run)."""
        self._tools = [
            {"name": str(t["name"]), "parameters": dict(t.get("parameters", {}) or {})}
            for t in tools
        ]
        self._tool_names = [str(t["name"]) for t in self._tools]
        self._tool_rounds = 0

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        tail = prompt[-2000:]
        if "TOOLS:" in prompt:
            if not self._tools:
                return f"FINAL: {tail.strip()[:400]}"
            if self._tool_rounds == 0:
                self._tool_rounds += 1
                first = self._tools[0]
                args = {k: "" for k in first["parameters"]}
                return f"TOOL: {first['name']} {json.dumps(args)}"
            return f"FINAL: used {self._tool_names[0]} offline; observation noted."
        if "PLAN:" in prompt:
            return (
                "1. analyse the goal\n"
                "2. gather information\n"
                "3. execute and verify\n"
                "GOAL_METABLE: yes"
            )
        if "REFLECT:" in prompt:
            return "CRITIQUE: previous attempt was sufficient.\nGOAL_MET: yes"
        return f"echo: {tail.strip()[:400]}"


class HttpOpenAICompat:
    """Client for any OpenAI-compatible /v1/chat/completions endpoint.

    Point ``base_url`` at the Silverwing serving bridge (e.g.
    ``http://localhost:8000/v1`` with ``model='silverwing-v2'``) once the
    freshly trained checkpoint is deployed.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "silverwing-v2",
        api_key: str = "local",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))
