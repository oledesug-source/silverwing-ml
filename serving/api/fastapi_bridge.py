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
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sw_platform.harness.agent import AgentResponse, HarnessConfig, PydanticAgentHarness

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Silverwing Agent Bridge",
    description="WebSocket/SSE bridge between the pydantic_ai agent harness and frontend",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Connection manager for WebSocket clients
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self) -> None:
        self._active_connections: dict[str, WebSocket] = {}
        self._session_harnesses: dict[str, PydanticAgentHarness] = {}

    async def connect(self, ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        self._active_connections[session_id] = ws

    def disconnect(self, session_id: str) -> None:
        self._active_connections.pop(session_id, None)
        harness = self._session_harnesses.pop(session_id, None)
        if harness:
            # Clean up session resources
            pass

    def get_connection(self, session_id: str) -> WebSocket | None:
        return self._active_connections.get(session_id)

    def create_session(
        self, session_id: str, config: HarnessConfig | None = None
    ) -> PydanticAgentHarness:
        """Create a new agent session with its own harness."""
        harness = PydanticAgentHarness(config or HarnessConfig())
        self._session_harnesses[session_id] = harness
        return harness

    def get_session(self, session_id: str) -> PydanticAgentHarness | None:
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
    response = harness.run(
        message.message,
        reset_history=message.reset_history,
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
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "error": "Invalid JSON"})
                continue

            action = data.get("action", "")
            if action == "message":
                content = data.get("content", "")

                # Stream the agent response
                t0 = time.monotonic()

                # Send typing indicator
                await ws.send_json({"type": "typing", "status": "started"})

                # Attempt to use the harness
                try:
                    async def run_agent(msg: str) -> AgentResponse:
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None, harness.run, msg
                        )
                        return response

                    response = await run_agent(content)

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


# Mount static files for the frontend
_frontend_static = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "static"
)
if os.path.isdir(_frontend_static):
    app.mount("/app", StaticFiles(directory=_frontend_static, html=True), name="frontend")
