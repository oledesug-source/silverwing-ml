"""Extended HTTP API for the Intelligence Runtime.

``IntelligenceHandler`` extends the existing ``SilverwingHandler`` with
new endpoints for chat, tool execution, and capability discovery.  The
legacy ``/generate``, ``/health``, and ``/info`` endpoints are preserved.
"""

from __future__ import annotations

import json

from serving.api.server import ApiResponse, SilverwingHandler


class IntelligenceHandler(SilverwingHandler):
    """Extended HTTP request handler adding intelligence endpoints.

    New endpoints:
        POST /chat          — orchestration loop (tool-use aware)
        POST /tool/execute  — direct single tool execution
        GET  /capabilities  — list registered capabilities

    Preserved endpoints:
        POST /generate      — raw text generation (legacy)
        GET  /health        — health check
        GET  /info           — model info
    """

    server_orchestrator = None  # Set by serve()
    server_registry = None  # Set by serve()

    def do_POST(self) -> None:
        if self.path == "/chat":
            self._handle_chat()
        elif self.path == "/tool/execute":
            self._handle_tool_execute()
        elif self.path == "/generate":
            super()._handle_generate()
        else:
            self._send_json(ApiResponse(success=False, error="Not found"), 404)

    def do_GET(self) -> None:
        if self.path == "/capabilities":
            self._handle_capabilities()
        elif self.path == "/health":
            self._send_json(ApiResponse(success=True, data={"status": "ok"}))
        elif self.path == "/info":
            self._handle_info()
        else:
            self._send_json(ApiResponse(success=False, error="Not found"), 404)

    # ------------------------------------------------------------------
    # New endpoints
    # ------------------------------------------------------------------

    def _handle_chat(self) -> None:
        """POST /chat — full orchestration loop with tool-use."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body) if body else {}

            message = request.get("message", "")
            max_rounds = request.get("max_rounds", 5)

            if not message:
                self._send_json(
                    ApiResponse(success=False, error="Missing 'message'"), 400
                )
                return

            if self.server_orchestrator is None:
                self._send_json(
                    ApiResponse(success=False, error="No orchestrator loaded"), 503
                )
                return

            from .orchestration import ChatRequest

            chat_request = ChatRequest(message=message, max_rounds=max_rounds)
            response = self.server_orchestrator.handle_request(chat_request)
            self._send_json(ApiResponse(success=True, data=response.to_dict()))

        except json.JSONDecodeError as exc:
            self._send_json(
                ApiResponse(success=False, error=f"Invalid JSON: {exc}"), 400
            )
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500
            )

    def _handle_tool_execute(self) -> None:
        """POST /tool/execute — execute a single tool directly."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body) if body else {}

            tool_name = request.get("tool", "")
            arguments = request.get("arguments", {})

            if not tool_name:
                self._send_json(
                    ApiResponse(success=False, error="Missing 'tool'"), 400
                )
                return

            if self.server_registry is None:
                self._send_json(
                    ApiResponse(success=False, error="No capability registry"), 503
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
                ApiResponse(success=False, error=f"Invalid JSON: {exc}"), 400
            )
        except Exception as exc:
            self._send_json(
                ApiResponse(success=False, error=str(exc)), 500
            )

    def _handle_capabilities(self) -> None:
        """GET /capabilities — list registered capabilities."""
        if self.server_registry is None:
            self._send_json(
                ApiResponse(success=False, error="No capability registry"), 503
            )
            return

        caps = [
            {
                "name": cap.name,
                "description": cap.description,
                "parameters": cap.parameters,
                "source": cap.source,
                "tags": cap.tags,
            }
            for cap in self.server_registry.list_capabilities()
        ]
        self._send_json(ApiResponse(success=True, data=caps))
