"""Model Context Protocol (MCP) server and client implementations.

Implements a subset of the MCP specification (https://spec.modelcontextprotocol.io/)
for connecting agents to context servers that provide tools, resources,
and prompts.

Provides:

    - ``MCPServer``: A lightweight MCP-compatible server that exposes
      tools, resources, and dynamic prompts to LLM agents.
    - ``MCPClient``: A client that connects to MCP servers and wraps their
      capabilities as callable tools.

The protocol uses JSON-RPC 2.0 over stdio or HTTP for transport.  This
implementation provides an in-process Python API for registering and
discovery capabilities without requiring an external MCP runtime.

Example server::

    server = MCPServer("my-tools")
    server.add_tool("search", description="Search documents",
                    handler=lambda query: {"results": []})
    response = server.call_tool("search", {"query": "hello"})

Example client::

    client = MCPClient()
    client.connect_stdio("python", "-m", "my_mcp_server")
    result = client.call_tool("search", {"query": "hello"})
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ToolHandler = Callable[..., Any]
ResourceHandler = Callable[[str], str | bytes]
PromptHandler = Callable[[dict[str, Any]], str]


@dataclass
class MCPTool:
    """An MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: ToolHandler | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema or {"type": "object", "properties": {}},
            "annotations": self.annotations,
        }


@dataclass
class MCPResource:
    """An MCP resource (URI-addressable data)."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"
    handler: ResourceHandler | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPrompt:
    """An MCP prompt template."""

    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    handler: PromptHandler | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class MCPServer:
    """A lightweight in-process MCP server.

    Supports registering tools, resources, and prompts, then serving
    them via JSON-RPC over stdio.  Can also be used directly in-process
    for testing and rapid prototyping.

    Args:
        name:    Server name (identifies the capability provider).
        version: Server version string.
        timeout: Default timeout for tool execution (seconds).
    """

    def __init__(self, name: str, version: str = "0.1.0", timeout: float = 30.0) -> None:
        self.name = name
        self.version = version
        self.timeout = timeout
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._request_id = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
        handler: ToolHandler | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> MCPServer:
        """Register a tool on the server.

        Args:
            name:        Tool name (must be unique).
            description: Human-readable description.
            input_schema: JSON Schema for the tool's arguments.
            handler:     Callable invoked when the tool is called.
            annotations: Optional metadata (e.g., title, destructiveFlag).

        Returns:
            self (for method chaining).
        """
        self._tools[name] = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema or {},
            handler=handler,
            annotations=annotations or {},
        )
        return self

    def add_resource(
        self,
        uri: str,
        name: str,
        description: str = "",
        mime_type: str = "text/plain",
        handler: ResourceHandler | None = None,
    ) -> MCPServer:
        """Register a resource on the server."""
        self._resources[uri] = MCPResource(
            uri=uri, name=name, description=description,
            mime_type=mime_type, handler=handler,
        )
        return self

    def add_prompt(
        self,
        name: str,
        description: str,
        arguments: list[dict[str, Any]] | None = None,
        handler: PromptHandler | None = None,
    ) -> MCPServer:
        """Register a prompt template on the server."""
        self._prompts[name] = MCPPrompt(
            name=name, description=description,
            arguments=arguments or [], handler=handler,
        )
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools.values())

    @property
    def resources(self) -> list[MCPResource]:
        return list(self._resources.values())

    @property
    def prompts(self) -> list[MCPPrompt]:
        return list(self._prompts.values())

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools as JSON-serializable dicts."""
        return [t.to_dict() for t in self._tools.values()]

    def list_resources(self) -> list[dict[str, Any]]:
        """Return all registered resources as JSON-serializable dicts."""
        return [r.to_dict() for r in self._resources.values()]

    def list_prompts(self) -> list[dict[str, Any]]:
        """Return all registered prompts as JSON-serializable dicts."""
        return [p.to_dict() for p in self._prompts.values()]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a tool and return the MCP-formatted result.

        Args:
            name:       Tool name.
            arguments:  Keyword arguments for the tool handler.

        Returns:
            MCP response dict with `content` and `isError` fields.
        """
        arguments = arguments or {}
        tool = self._tools.get(name)
        if tool is None:
            return {"content": [], "isError": True,
                    "error": f"Tool '{name}' not found"}
        if tool.handler is None:
            return {"content": [], "isError": True,
                    "error": f"Tool '{name}' has no handler"}

        try:
            result = tool.handler(**arguments)
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "json", "json": result}]
            elif isinstance(result, list):
                content = result
            else:
                content = [{"type": "text", "text": str(result)}]
            return {"content": content, "isError": False}
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            return {"content": [{"type": "text", "text": str(exc)}],
                    "isError": True}

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI."""
        resource = self._resources.get(uri)
        if resource is None:
            return {"contents": [], "isError": True,
                    "error": f"Resource '{uri}' not found"}
        if resource.handler is None:
            return {"contents": [], "isError": True,
                    "error": f"Resource '{uri}' has no handler"}
        try:
            data = resource.handler(uri)
            return {
                "contents": [{"uri": uri, "mimeType": resource.mime_type,
                              "text": data if isinstance(data, str) else str(data)}],
                "isError": False,
            }
        except Exception as exc:
            return {"contents": [], "isError": True, "error": str(exc)}

    def render_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render a prompt template with given arguments."""
        arguments = arguments or {}
        prompt = self._prompts.get(name)
        if prompt is None:
            return {"messages": [], "isError": True,
                    "error": f"Prompt '{name}' not found"}
        if prompt.handler is None:
            return {"messages": [], "isError": True,
                    "error": f"Prompt '{name}' has no handler"}
        try:
            result = prompt.handler(arguments)
            return {"messages": [{"role": "user", "content": result}],
                    "isError": False}
        except Exception as exc:
            return {"messages": [], "isError": True, "error": str(exc)}

    # ------------------------------------------------------------------
    # JSON-RPC over stdio
    # ------------------------------------------------------------------

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single JSON-RPC 2.0 request and return a response."""
        req_id = request.get("id", None)

        if request.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32600, "message": "Invalid Request"}}

        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            self._initialized = True
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": True},
                        "prompts": {"listChanged": True},
                    },
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            }

        if not self._initialized and method != "initialize":
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": "Server not initialized"}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"tools": self.list_tools()}}

        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": self.call_tool(name, arguments)}

        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"resources": self.list_resources()}}

        if method == "resources/read":
            uri = params.get("uri", "")
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": self.read_resource(uri)}

        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"prompts": self.list_prompts()}}

        if method == "prompts/render":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": self.render_prompt(name, arguments)}

        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"}}

    def run_stdio(self) -> None:
        """Run the MCP server over stdio (JSON-RPC 2.0).

        Reads JSON-RPC requests from stdin, writes responses to stdout.
        This is the standard MCP transport for local tool servers.
        """
        logger.info("MCP server '%s' v%s starting on stdio", self.name, self.version)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }), flush=True)
            except Exception as exc:
                logger.exception("Unhandled error processing request")
                print(json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32603, "message": str(exc)},
                }), flush=True)


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------

class MCPClient:
    """MCP client that can connect to servers over stdio or in-process.

    Provides a unified interface for discovering and invoking tools
    from any MCP-compatible server.

    Example::

        client = MCPClient()
        client.connect_in_process(server)
        result = client.call_tool("search", {"query": "hello"})
    """

    def __init__(self) -> None:
        self._server: MCPServer | None = None
        self._stdio_process: subprocess.Popen | None = None
        self._connected = False
        self._tools: dict[str, MCPTool] = {}

    def connect_in_process(self, server: MCPServer) -> None:
        """Connect to an in-process MCPServer instance."""
        self._server = server
        self._connected = True
        self._tools = {t.name: t for t in server.tools}
        logger.info("Connected to in-process MCP server: %s", server.name)

    def connect_stdio(
        self,
        command: str,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> None:
        """Connect to an MCP server running over stdio.

        Args:
            command:  Executable command (e.g., "python").
            args:     Additional arguments (e.g., "-m", "my_mcp_server").
            env:      Optional environment variables.
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        self._stdio_process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
        )
        self._connected = True

        # Initialize
        self._send_stdio({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2024-11-05",
                                      "clientInfo": {"name": "mcp-client", "version": "0.1.0"},
                                      "capabilities": {}}})

        # List tools
        resp = self._send_stdio({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        for t in resp.get("result", {}).get("tools", []):
            self._tools[t["name"]] = MCPTool(
                name=t["name"], description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )

        logger.info("Connected to %d tools via stdio", len(self._tools))

    def _send_stdio(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request over stdio and return the response."""
        if self._stdio_process is None:
            raise RuntimeError("Not connected to a stdio server")

        assert self._stdio_process.stdin is not None
        assert self._stdio_process.stdout is not None

        self._stdio_process.stdin.write(json.dumps(request) + "\n")
        self._stdio_process.stdin.flush()

        # Read one line response
        line = self._stdio_process.stdout.readline()
        if not line:
            raise RuntimeError("Server closed connection")
        return json.loads(line.strip())

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools from connected servers."""
        if self._server is not None:
            return self._server.list_tools()
        return [t.to_dict() for t in self._tools.values()]

    def list_resources(self) -> list[dict[str, Any]]:
        """List all available resources."""
        if self._server is not None:
            return self._server.list_resources()
        resp = self._send_stdio({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
        return resp.get("result", {}).get("resources", [])

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool by name with given arguments.

        Args:
            name:       Tool name.
            arguments:  Arguments dict matching the tool's input schema.

        Returns:
            MCP tool response dict with `content` and `isError` fields.
        """
        arguments = arguments or {}
        if self._server is not None:
            return self._server.call_tool(name, arguments)
        if name not in self._tools:
            return {"content": [], "isError": True,
                    "error": f"Tool '{name}' not found"}
        resp = self._send_stdio({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        if "error" in resp:
            return {"content": [], "isError": True, "error": resp["error"]}
        return resp.get("result", {"content": [], "isError": False})

    def disconnect(self) -> None:
        """Disconnect from the server and clean up resources."""
        if self._stdio_process is not None:
            self._stdio_process.terminate()
            self._stdio_process.wait(timeout=5)
            self._stdio_process = None
        self._server = None
        self._connected = False
        self._tools.clear()
