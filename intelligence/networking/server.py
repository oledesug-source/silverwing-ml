"""Lightweight HTTP server for ML model serving.

Provides a base HTTP server that can be extended to serve model
predictions, health checks, and management endpoints.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


@dataclass
class Request:
    """Parsed HTTP request."""

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""
    query: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse body as JSON."""
        if not self.body:
            return {}
        return json.loads(self.body)


@dataclass
class Response:
    """HTTP response to send."""

    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> bytes:
        """Serialize body to JSON bytes."""
        return json.dumps(self.body).encode()


class _RequestHandler(BaseHTTPRequestHandler):
    """Internal handler that delegates to the Server."""

    server_ref: Server | None = None

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_PUT(self) -> None:
        self._handle("PUT")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def _handle(self, method: str) -> None:
        if self.server_ref is None:
            self.send_error(500)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length else ""

        req = Request(
            method=method,
            path=self.path,
            headers=dict(self.headers),
            body=body,
        )

        resp = self.server_ref.handle(req)

        self.send_response(resp.status)
        for k, v in resp.headers.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.to_json())

    def log_message(self, format: str, *args: Any) -> None:
        pass


class Server:
    """Simple HTTP server for ML endpoints.

    Usage::

        server = Server(host="0.0.0.0", port=8000)

        @server.route("GET", "/health")
        def health(req):
            return Response(body={"status": "ok"})

        @server.route("POST", "/predict")
        def predict(req):
            data = req.json()
            result = model(data["input"])
            return Response(body={"prediction": result})

        server.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._host = host
        self._port = port
        self._routes: dict[tuple[str, str], Callable[[Request], Response]] = {}
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

        self._routes[("GET", "/health")] = lambda req: Response(
            body={"status": "ok"}
        )

    def route(
        self, method: str, path: str
    ) -> Callable[[Callable[[Request], Response]], Callable[[Request], Response]]:
        """Decorator to register a route handler."""
        def decorator(
            fn: Callable[[Request], Response]
        ) -> Callable[[Request], Response]:
            self._routes[(method.upper(), path)] = fn
            return fn
        return decorator

    def handle(self, req: Request) -> Response:
        """Route a request to its handler."""
        handler = self._routes.get((req.method, req.path))
        if handler:
            return handler(req)
        return Response(status=404, body={"error": "Not found"})

    def start(self, blocking: bool = False) -> None:
        """Start the server."""
        _RequestHandler.server_ref = self
        self._httpd = HTTPServer(
            (self._host, self._port), _RequestHandler
        )
        if blocking:
            self._httpd.serve_forever()
        else:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the server."""
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"
