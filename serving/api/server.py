"""HTTP API for serving Silverwing models.

Provides a lightweight REST API using Python's built-in http.server
(no external dependencies required).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any


@dataclass
class ApiResponse:
    """Standard API response envelope."""

    success: bool
    data: Any = None
    error: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }, ensure_ascii=False, indent=2)


class SilverwingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Silverwing inference.

    Endpoints:
        POST /generate   — generate text from a prompt
        GET  /health     — health check
        GET  /info       — model info
    """

    server_model = None  # Set by serve()
    server_config = None

    def do_POST(self) -> None:
        if self.path == "/generate":
            self._handle_generate()
        else:
            self._send_json(ApiResponse(success=False, error="Not found"), 404)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(ApiResponse(success=True, data={"status": "ok"}))
        elif self.path == "/info":
            self._handle_info()
        else:
            self._send_json(ApiResponse(success=False, error="Not found"), 404)

    def _handle_generate(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body) if body else {}

            prompt = request.get("prompt", "")
            max_new_tokens = request.get("max_new_tokens", 128)
            temperature = request.get("temperature", 0.0)

            if not prompt:
                self._send_json(ApiResponse(success=False, error="Missing 'prompt'"), 400)
                return

            if self.server_model is None:
                self._send_json(ApiResponse(success=False, error="No model loaded"), 503)
                return

            result = self.server_model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            self._send_json(ApiResponse(
                success=True,
                data={
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "prompt": prompt,
                },
            ))
        except json.JSONDecodeError as exc:
            self._send_json(ApiResponse(success=False, error=f"Invalid JSON: {exc}"), 400)
        except Exception as exc:
            self._send_json(ApiResponse(success=False, error=str(exc)), 500)

    def _handle_info(self) -> None:
        if self.server_model is None:
            self._send_json(ApiResponse(success=False, error="No model loaded"), 503)
            return
        self._send_json(ApiResponse(
            success=True,
            data={
                "model_id": getattr(self.server_model, "_model_id", "unknown"),
                "max_new_tokens": getattr(self.server_model, "_max_new_tokens", 128),
            },
        ))

    def _send_json(self, response: ApiResponse, status: int = 200) -> None:
        body = response.to_json().encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default logging
