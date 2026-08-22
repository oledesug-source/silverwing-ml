"""FastAPI WebSocket bridge — streams agent harness to the frontend.

Layer 2 Bridge: Connects the pydantic_ai agent harness to the frontend
via WebSocket and Server-Sent Events (SSE).  Streams terminal logs,
UI states, and agent responses in real time.

Run with::

    uvicorn serving.api.fastapi_bridge:app --reload --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sw_platform.harness.agent import HarnessConfig, PydanticAgentHarness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime tuning (environment-configurable)
# ---------------------------------------------------------------------------

# Bounded executor for blocking harness calls.  Using the default loop
# executor lets concurrent agent sessions starve each other; a dedicated
# bounded pool isolates them and caps resource usage.
_EXECUTOR_WORKERS = int(os.environ.get("SILVERWING_BRIDGE_WORKERS", "4"))
_executor = ThreadPoolExecutor(
    max_workers=_EXECUTOR_WORKERS, thread_name_prefix="sw-harness"
)

# Maximum inbound WebSocket message size, in bytes.
_MAX_WS_MESSAGE_BYTES = int(os.environ.get("SILVERWING_BRIDGE_MAX_MESSAGE", "65536"))

# Maximum chat message length accepted over REST/SSE, in characters.
_MAX_CHAT_MESSAGE_CHARS = int(os.environ.get("SILVERWING_BRIDGE_MAX_CHARS", "32000"))

# Agent sessions idle longer than this (with no live connection) are evicted
# to bound memory usage on long-running servers.
_SESSION_TTL_SECONDS = float(os.environ.get("SILVERWING_BRIDGE_SESSION_TTL", "3600"))

app = FastAPI(
    title="Silverwing Agent Bridge",
    description="WebSocket/SSE bridge between the pydantic_ai agent harness and frontend",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Connection manager for WebSocket clients
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections per session.

    Session harnesses are cleaned up on disconnect and idle sessions are
    evicted after ``_SESSION_TTL_SECONDS`` to avoid unbounded memory growth.
    """

    def __init__(self) -> None:
        self._active_connections: dict[str, WebSocket] = {}
        self._session_harnesses: dict[str, PydanticAgentHarness] = {}
        self._last_activity: dict[str, float] = {}

    async def connect(self, ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        old = self._active_connections.get(session_id)
        if old is not None and old is not ws:
            # A stale connection exists for this session — close it so the
            # new client takes over cleanly.
            self._active_connections.pop(session_id, None)
            try:
                await old.close(code=4000, reason="Replaced by new connection")
            except Exception:
                pass
        self._active_connections[session_id] = ws
        self.touch(session_id)

    def touch(self, session_id: str) -> None:
        """Record activity for TTL-based session eviction."""
        self._last_activity[session_id] = time.monotonic()

    def disconnect(self, session_id: str) -> None:
        self._active_connections.pop(session_id, None)
        self._last_activity.pop(session_id, None)
        harness = self._session_harnesses.pop(session_id, None)
        if harness is not None:
            harness.reset()
            logger.info(
                "Cleaned up session %s (%d session(s) remaining)",
                session_id,
                len(self._session_harnesses),
            )

    def evict_idle_sessions(self) -> int:
        """Drop harnesses for sessions with no live connection past the TTL."""
        now = time.monotonic()
        idle = [
            sid
            for sid, harness in self._session_harnesses.items()
            if sid not in self._active_connections
            and now - self._last_activity.get(sid, now) > _SESSION_TTL_SECONDS
        ]
        for sid in idle:
            logger.info("Evicting idle session %s", sid)
            self.disconnect(sid)
        return len(idle)

    def get_connection(self, session_id: str) -> WebSocket | None:
        return self._active_connections.get(session_id)

    def create_session(
        self, session_id: str, config: HarnessConfig | None = None
    ) -> PydanticAgentHarness:
        """Create a new agent session with its own harness."""
        self.evict_idle_sessions()
        harness = PydanticAgentHarness(config or HarnessConfig())
        self._session_harnesses[session_id] = harness
        self._last_activity[session_id] = time.monotonic()
        return harness

    def get_session(self, session_id: str) -> PydanticAgentHarness | None:
        self.touch(session_id)
        return self._session_harnesses.get(session_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all connected clients."""
        dead = []
        for sid, ws in self._active_connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(sid)
        for sid in dead:
            self.disconnect(sid)

    async def send_to_session(self, session_id: str, message: dict[str, Any]) -> bool:
        """Send a message to a specific session."""
        ws = self._active_connections.get(session_id)
        if ws is None:
            return False
        try:
            await ws.send_json(message)
            return True
        except Exception:
            self.disconnect(session_id)
            return False


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Incoming chat message from the frontend."""

    message: str = Field(..., description="User message to send to the agent")
    model: str = Field(default="openai:gpt-4o")
    max_rounds: int = Field(default=5)
    reset_history: bool = Field(default=False)


class ChatResponse(BaseModel):
    """Structured chat response."""

    session_id: str
    message: str
    response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str = ""


class ToolListResponse(BaseModel):
    """Response listing available tools."""

    tools: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    """Serve the frontend."""
    frontend_index = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "static", "index.html"
    )
    # Try frontend static dir first, fall back to a generated page
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    # Return a minimal health/info JSON if no frontend
    return JSONResponse({"service": "Silverwing Agent Bridge", "status": "running"})


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "Silverwing Agent Bridge",
        "timestamp": time.time(),
    })


@app.get("/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """List all available tools in the harness."""
    harness = PydanticAgentHarness()
    tools = [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "risk_level": t.risk_level,
            "permission_required": t.permission_required,
            "tags": t.tags,
        }
        for t in harness.tools
    ]
    return ToolListResponse(tools=tools)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
) -> ChatResponse:
    """Synchronous chat endpoint — process message and return response."""
    session_id = f"rest-{uuid.uuid4().hex[:8]}"
    config = HarnessConfig(
        model=message.model,
        max_rounds=message.max_rounds,
    )
    harness = PydanticAgentHarness(config)

    t0 = time.monotonic()
    if len(message.message) > _MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Message exceeds {_MAX_CHAT_MESSAGE_CHARS} characters",
        )
    response = await asyncio.to_thread(
        harness.run, message.message, None, message.reset_history
    )
    elapsed = time.monotonic() - t0

    return ChatResponse(
        session_id=session_id,
        message=message.message,
        response=response.text,
        tool_calls=[tc.to_dict() for tc in response.tool_calls],
        elapsed_seconds=elapsed,
        success=response.success,
        error=response.error,
    )


@app.get("/sessions/{session_id}/tools")
async def session_tools(session_id: str) -> ToolListResponse:
    """List tools for a specific session."""
    harness = manager.get_session(session_id)
    if harness is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    tools = [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "risk_level": t.risk_level,
            "permission_required": t.permission_required,
            "tags": t.tags,
        }
        for t in harness.tools
    ]
    return ToolListResponse(tools=tools)


# ---------------------------------------------------------------------------
# WebSocket endpoint — streams agent output in real time
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time agent communication.

    Protocol:
        Client sends JSON: {"action": "message", "content": "..."}
        Server sends JSON: {"type": "response|tool_call|audit|error", ...}

    The server streams:
        - Agent response text (chunked)
        - Tool call records (name, args, result)
        - Audit events (permission checks, execution trace)
        - Error messages
    """
    # Create or reuse session harness
    harness = manager.get_session(session_id)
    if harness is None:
        harness = manager.create_session(session_id)

    await manager.connect(ws, session_id)
    await ws.send_json({
        "type": "session_ready",
        "session_id": session_id,
        "tools": [t.name for t in harness.tools],
    })

    try:
        while True:
            raw = await ws.receive_text()
            if len(raw.encode("utf-8")) > _MAX_WS_MESSAGE_BYTES:
                await ws.send_json({"type": "error", "error": "Message too large"})
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "error": "Invalid JSON"})
                continue

            manager.touch(session_id)
            action = data.get("action", "")
            if action == "message":
                content = data.get("content", "")

                # Stream the agent response
                t0 = time.monotonic()

                # Send typing indicator
                await ws.send_json({"type": "typing", "status": "started"})

                # Attempt to use the harness
                try:
                    response = await asyncio.to_thread(harness.run, content)

                    await ws.send_json({
                        "type": "response",
                        "text": response.text,
                        "elapsed_seconds": time.monotonic() - t0,
                        "success": response.success,
                    })

                    for tc in response.tool_calls:
                        await ws.send_json({
                            "type": "tool_call",
                            "data": tc.to_dict(),
                        })

                    # Send audit log entries
                    for event in harness.audit_log[-10:]:
                        await ws.send_json({
                            "type": "audit",
                            "data": event,
                        })

                except Exception as exc:
                    await ws.send_json({
                        "type": "error",
                        "error": str(exc),
                    })
                finally:
                    await ws.send_json({"type": "typing", "status": "ended"})

            elif action == "list_tools":
                await ws.send_json({
                    "type": "tools",
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                            "risk_level": t.risk_level,
                        }
                        for t in harness.tools
                    ],
                })

            elif action == "ping":
                await ws.send_json({"type": "pong"})

            elif action == "reset":
                if harness._conversation_history:
                    harness._conversation_history.clear()
                await ws.send_json({"type": "reset_done"})

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info("WebSocket client disconnected: %s", session_id)


# ---------------------------------------------------------------------------
# Structured request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log every HTTP request with method, path, status, and latency."""
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "%s %s -> 500 (%.1f ms)",
            request.method,
            request.url.path,
            (time.perf_counter() - t0) * 1000,
        )
        raise
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
    )
    return response


# ---------------------------------------------------------------------------
# SSE streaming chat — event-level streaming endpoint
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict[str, Any]) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/chat/stream")
async def chat_stream(message: ChatMessage) -> StreamingResponse:
    """Stream a chat interaction over Server-Sent Events.

    Emits ``start`` → ``tool_call``* → ``response`` → ``done`` frames so the
    frontend renders progress incrementally.  Token-level streaming will be
    layered on once the underlying provider exposes a streaming API; today
    the harness returns complete responses.
    """
    if len(message.message) > _MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Message exceeds {_MAX_CHAT_MESSAGE_CHARS} characters",
        )

    session_id = f"sse-{uuid.uuid4().hex[:8]}"
    harness = PydanticAgentHarness(
        HarnessConfig(model=message.model, max_rounds=message.max_rounds)
    )

    async def event_stream() -> Any:
        yield _sse("start", {"session_id": session_id})
        try:
            response = await asyncio.to_thread(
                harness.run, message.message, None, message.reset_history
            )
            for tc in response.tool_calls:
                yield _sse("tool_call", tc.to_dict())
            yield _sse("response", {
                "session_id": session_id,
                "text": response.text,
                "success": response.success,
                "error": response.error,
                "elapsed_seconds": response.elapsed_seconds,
            })
        except Exception as exc:
            logger.exception("SSE chat failed for session %s", session_id)
            yield _sse("error", {"error": str(exc)})
        finally:
            yield _sse("done", {"session_id": session_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Mount static files for the frontend
_frontend_static = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "static"
)
if os.path.isdir(_frontend_static):
    app.mount("/app", StaticFiles(directory=_frontend_static, html=True), name="frontend")
