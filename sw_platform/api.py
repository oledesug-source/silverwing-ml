"""Extended HTTP API for the Controlled Intelligence Platform.

``PlatformHandler`` extends ``SilverwingHandler`` with new endpoints for
chat, tool execution, capability discovery, and gesture control.  The
legacy endpoints are preserved.
"""

from __future__ import annotations

import json
import time
import uuid
from urllib.parse import urlparse

from serving.api.server import ApiResponse, SilverwingHandler


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten an OpenAI-style message list into the SFT training format.

    Silverwing Decoder V2 was instruction-tuned on
    ``Question: {instruction}\\nAnswer: {response}`` pairs, so the
    conversation is flattened to that template (user turns become
    Questions, assistant turns become Answers).
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if not content:
            continue
        if role == "system":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Answer: {content}")
        else:
            parts.append(f"Question: {content}")
    if parts and parts[-1].startswith("Question:"):
        # Trailing space matters: a bare "Answer:" tail makes the SFT model
        # emit EOS immediately; "Answer: " anchors continuation correctly.
        parts.append("Answer: ")
    return "\n".join(parts)


class PlatformHandler(SilverwingHandler):
    """HTTP request handler adding platform intelligence endpoints.

    New endpoints:
        POST /v1/chat              — orchestration loop (tool-use aware)
        POST /v1/chat/completions  — OpenAI-compatible chat completions
        POST /v1/tools/execute     — direct single tool execution
        GET  /v1/capabilities      — list registered capabilities
        GET  /v1/gestures          — static gesture → action mapping table
        GET  /v1/gestures/status   — subsystem availability + config snapshot
        GET  /v1/gestures/stats    — live system metrics (CPU, mem, battery)

    Preserved endpoints:
        POST /generate      — raw text generation (legacy)
        GET  /health        — health check
        GET  /info          — model info
    """

    server_orchestrator = None  # Set by serve()
    server_registry = None  # Set by serve()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/chat":
            self._handle_chat()
        elif path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/v1/tools/execute":
            self._handle_tool_execute()
        elif path == "/generate":
            super()._handle_generate()
        else:
            self._send_json(ApiResponse(success=False, error="Not found"), 404)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/capabilities":
            self._handle_capabilities()
        elif path == "/health":
            self._send_json(ApiResponse(success=True, data={"status": "ok"}))
        elif path == "/info":
            self._handle_info()
        elif path == "/v1/gestures":
            self._handle_gestures()
        elif path == "/v1/gestures/status":
            self._handle_gestures_status()
        elif path == "/v1/gestures/stats":
            self._handle_gestures_stats()
        else:
            self._send_json(ApiResponse(success=False, error="Not found"), 404)

    def _handle_chat(self) -> None:
        """POST /v1/chat — full orchestration loop with tool-use."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body) if body else {}

            message = request.get("message", "")
            max_rounds = request.get("max_rounds", 5)

            if not message:
                self._send_json(
                    ApiResponse(success=False, error="Missing 'message'"), 400,
                )
                return

            if self.server_orchestrator is None:
                self._send_json(
                    ApiResponse(success=False, error="No orchestrator loaded"), 503,
                )
                return

            from sw_platform.orchestration.orchestrator import ChatRequest
            chat_request = ChatRequest(message=message, max_rounds=max_rounds)
            response = self.server_orchestrator.handle_request(chat_request)
            self._send_json(ApiResponse(success=True, data=response.to_dict()))

        except json.JSONDecodeError as exc:
            self._send_json(
                ApiResponse(success=False, error=f"Invalid JSON: {exc}"), 400,
            )
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500,
            )

    def _handle_chat_completions(self) -> None:
        """POST /v1/chat/completions — OpenAI-compatible chat completions.

        Backed directly by the Layer 4 ModelProvider (no orchestration loop),
        so any OpenAI-SDK-compatible client (pydantic_ai, openai python
        package, curl) can use the served Silverwing model.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body) if body else {}

            messages = request.get("messages") or []
            if not messages:
                self._send_json(
                    ApiResponse(success=False,
                                error="'messages' must be a non-empty list"), 400,
                )
                return
            if request.get("stream"):
                self._send_json(
                    ApiResponse(success=False,
                                error="stream=true is not supported yet"), 400,
                )
                return

            orch = self.server_orchestrator
            if orch is None or getattr(orch, "generator", None) is None:
                self._send_json(
                    ApiResponse(success=False, error="No model provider loaded"), 503,
                )
                return

            from silverwing_platform.models import GenerationConfig, InferenceRequest

            config = GenerationConfig(
                max_new_tokens=int(request.get("max_tokens", 128)),
                temperature=float(request.get("temperature", 0.7)),
                top_p=float(request.get("top_p", 0.9)),
            )
            prompt = _messages_to_prompt(messages)
            response = orch.generator.infer(InferenceRequest(prompt=prompt, config=config))

            model_name = request.get("model") or response.model_id or "silverwing-v2"
            usage = response.usage
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["generated_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
            }
            body_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        except json.JSONDecodeError as exc:
            self._send_json(
                ApiResponse(success=False, error=f"Invalid JSON: {exc}"), 400,
            )
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500,
            )

    def _handle_tool_execute(self) -> None:
        """POST /v1/tools/execute — execute a single tool directly."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body) if body else {}

            tool_name = request.get("tool", "")
            arguments = request.get("arguments", {})

            if not tool_name:
                self._send_json(
                    ApiResponse(success=False, error="Missing 'tool'"), 400,
                )
                return

            if self.server_registry is None:
                self._send_json(
                    ApiResponse(success=False, error="No capability registry"), 503,
                )
                return

            from intelligence.tools.protocol import ToolCall
            call = ToolCall(
                tool_name=tool_name,
                arguments=",".join(f"{k}={v}" for k, v in arguments.items()),
            )
            result = self.server_registry.execute_call(call)
            self._send_json(ApiResponse(
                success=True,
                data={
                    "tool": result.tool_name,
                    "output": result.output,
                    "success": result.success,
                    "error": result.error,
                },
            ))

        except json.JSONDecodeError as exc:
            self._send_json(
                ApiResponse(success=False, error=f"Invalid JSON: {exc}"), 400,
            )
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500,
            )

    def _handle_capabilities(self) -> None:
        """GET /v1/capabilities — list registered capabilities."""
        if self.server_registry is None:
            self._send_json(
                ApiResponse(success=False, error="No capability registry"), 503,
            )
            return

        caps = [
            {
                "name": cap.name,
                "version": cap.version,
                "description": cap.description,
                "input_schema": cap.input_schema,
                "risk_level": cap.risk_level,
                "enabled": cap.enabled,
                "tags": cap.tags,
            }
            for cap in self.server_registry.list(enabled_only=False)
        ]
        self._send_json(ApiResponse(success=True, data=caps))

    # ------------------------------------------------------------------
    # Gesture OS endpoints
    # ------------------------------------------------------------------

    def _handle_gestures(self) -> None:
        """GET /v1/gestures — static gesture → action mapping table."""
        try:
            from sw_platform.tools.gesture import get_gesture_registry
            data = get_gesture_registry()
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500,
            )
            return

        self._send_json(ApiResponse(
            success=True,
            data={
                "gestures": data["gestures"],
            },
        ))

    def _handle_gestures_status(self) -> None:
        """GET /v1/gestures/status — subsystem availability + config."""
        try:
            from sw_platform.tools.gesture import get_gesture_registry
            data = get_gesture_registry()
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500,
            )
            return

        self._send_json(ApiResponse(
            success=True,
            data=data["status"],
        ))

    def _handle_gestures_stats(self) -> None:
        """GET /v1/gestures/stats — live system metrics."""
        try:
            from sw_platform.tools.gesture import GestureCapabilityProvider
            provider = GestureCapabilityProvider()
            stats = provider.get_system_stats()
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500,
            )
            return

        self._send_json(ApiResponse(success=True, data=stats))
