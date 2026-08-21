"""Launch the SilverWing Platform with a web frontend.

Usage::

    python scripts/serve_platform.py              # default port 8000
    python scripts/serve_platform.py --port 9000  # custom port

Opens a browser-ready interface at http://localhost:<port>/ with:
  - Left panel: registered capabilities
  - Center: chat orchestration loop (LLM proposes, runtime decides)
  - Right: direct tool execution + audit trail
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sw_platform.api import PlatformHandler
from sw_platform.capabilities.registry import CapabilityRegistry
from sw_platform.orchestration.orchestrator import Orchestrator

FRONTEND_DIR = PROJECT_ROOT / "silverwing_platform" / "frontend"

# --- Built-in tools (safe, no torch) ---

_builtin_tools = {}


def _calculator(expression: str = "") -> str:
    """AST-based safe math evaluator."""
    import ast
    import operator as op

    ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
        ast.Pow: op.pow, ast.USub: op.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    if not expression:
        return "Error: empty expression"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _read_file(path: str = "") -> str:
    """Read-only file reader (64KB limit)."""
    if not path:
        return "Error: path is required"
    try:
        p = Path(path).resolve()
        if p.is_dir():
            return f"Error: '{path}' is a directory"
        if p.stat().st_size > 64 * 1024:
            return "Error: file too large (64KB limit)"
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"


def _list_dir(path: str = ".") -> str:
    """List directory contents."""
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for e in entries:
            prefix = "d " if e.is_dir() else "  "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except FileNotFoundError:
        return f"Error: path not found: {path}"


_builtin_tools["calculator"] = {
    "fn": _calculator,
    "description": "Safe math evaluator. Supports +, -, *, /, //, %, **, parentheses. No eval().",
    "parameters": {"expression": {"type": "string", "description": "math expression to evaluate"}},
    "tags": ["math", "calc", "compute"],
}
_builtin_tools["read_file"] = {
    "fn": _read_file,
    "description": "Read-only file reader (64KB limit). Cannot read directories or large files.",
    "parameters": {"path": {"type": "string", "description": "file path to read"}},
    "tags": ["file", "read", "io"],
}
_builtin_tools["list_dir"] = {
    "fn": _list_dir,
    "description": "List directory contents. Shows files and subdirectories.",
    "parameters": {"path": {"type": "string", "description": "directory path (default: current dir)"}},
    "tags": ["file", "list", "directory"],
}


def _select_model_provider(force_mock: bool = False):
    """Pick the Layer 4 model provider.

    Uses the real Silverwing Generator when the configured checkpoint exists,
    otherwise falls back to a deterministic MockProvider so the platform still
    boots (with a loud warning) for UI/tool development.
    """
    from silverwing_platform.models import GeneratorProvider, MockProvider

    if force_mock:
        print("  Model provider: MockProvider (--mock)")
        return MockProvider()

    try:
        checkpoint = None
        cfg_path = PROJECT_ROOT / "configs" / "inference.yaml"
        if cfg_path.exists():
            import yaml

            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            checkpoint = (raw.get("inference") or {}).get("checkpoint_path")
        ckpt = Path(checkpoint) if checkpoint else PROJECT_ROOT / "experiments/checkpoints/best.pt"
        if not ckpt.is_absolute():
            ckpt = PROJECT_ROOT / ckpt
        if ckpt.exists():
            print(f"  Model provider: GeneratorProvider ({ckpt})")
            return GeneratorProvider()
        print(f"  [warn] No checkpoint at {ckpt} - using MockProvider fallback")
        return MockProvider()
    except Exception as exc:
        print(f"  [warn] Provider selection failed ({exc}) - using MockProvider")
        return MockProvider()


def setup_platform(force_mock: bool = False):
    """Create and configure the platform with full Layer 1–4 integration.

    Wires together:
      - Built-in safe tools (calculator, file reader, dir listing)
      - MLOps capabilities (MLflow, W&B, KFP, Spark)
      - Layer 4 PolicyEngine (allow/deny/require_approval)
      - Layer 4 ApprovalManager (pending request lifecycle)
      - Layer 4 PlatformDatabase (SQLite persistence)
      - Layer 4 ModelProvider (lazy-loaded foundation.Generator)
    """
    from silverwing_platform.approvals import ApprovalManager
    from silverwing_platform.database import PlatformDatabase
    from silverwing_platform.policies import PolicyEngine
    from sw_platform.capabilities.schema import CapabilitySchema
    from sw_platform.permissions.policy import PermissionLevel

    registry = CapabilityRegistry()

    # Register built-in safe tools
    for name, info in _builtin_tools.items():
        schema = CapabilitySchema(
            name=name,
            description=info["description"],
            input_schema=info["parameters"],
            fn=info["fn"],
            source="builtin",
            tags=info["tags"],
            risk_level="low",
            permissions_required=[PermissionLevel.L0.name],
            timeout_seconds=10,
        )
        registry.register(schema)

    # Register MLOps capabilities (auto-discovers available tools)
    try:
        from sw_platform.tools.mlops import register_mlops_capabilities
        register_mlops_capabilities(registry)
    except Exception as exc:
        print(f"  [warn] MLOps capabilities not registered: {exc}")

    # Layer 4: database for persistence
    db = PlatformDatabase()

    # Layer 4: policy engine + approval manager
    policy_engine = PolicyEngine()
    approval_mgr = ApprovalManager(db=db)

    # Layer 4: model provider (lazy — no torch needed to start)
    generator = _select_model_provider(force_mock=force_mock)

    orchestrator = Orchestrator(
        registry=registry,
        generator=generator,
        policies=policy_engine,
        approvals=approval_mgr,
        database=db,
        max_steps=5,
    )
    return registry, orchestrator


class PlatformServerHandler(PlatformHandler):
    """Extended handler that also serves static frontend files."""

    server_registry = None
    server_orchestrator = None
    server_frontend_dir = None

    def do_GET(self):
        path = urlparse(self.path).path

        # API routes
        if path in ("/health", "/info", "/v1/capabilities"):
            super().do_GET()
            return

        # Serve frontend
        self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path in ("/v1/chat", "/v1/chat/completions", "/v1/tools/execute", "/generate"):
            super().do_POST()
        else:
            body = json.dumps({"success": False, "error": "Not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _serve_static(self, path):
        if path == "/":
            path = "/index.html"

        file_path = self.server_frontend_dir / path.lstrip("/")
        if not file_path.exists() or not file_path.is_file():
            # Serve index.html for SPA-style routing
            file_path = self.server_frontend_dir / "index.html"

        try:
            content = file_path.read_bytes()
            ext = file_path.suffix.lower()
            ct = {
                ".html": "text/html",
                ".js": "text/javascript",
                ".css": "text/css",
                ".json": "application/json",
                ".png": "image/png",
                ".svg": "image/svg+xml",
            }.get(ext, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception:
            self.send_error(404)

    def log_message(self, format, *args):
        # Show request logging
        if args:
            method = args[0].split()[0] if " " in args[0] else ""
            status = args[1] if len(args) > 1 else ""
            if method in ("GET", "POST"):
                color = "\033[32m" if str(status).startswith("2") else "\033[33m"
                reset = "\033[0m"
                print(f"  {color}{status}{reset} {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="SilverWing Platform Server")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--mock", action="store_true", help="Force MockProvider (skip model load)")
    args = parser.parse_args()

    print("\n  SilverWing Platform")
    print("  ===================\n")

    print("  Setting up platform...")
    registry, orchestrator = setup_platform(force_mock=args.mock)
    caps = registry.list(enabled_only=True)
    print(f"  {len(caps)} capabilities registered: {', '.join(c.name for c in caps)}")
    print(f"  Policy engine: {'active' if orchestrator.policies else 'inactive'}")
    print(f"  Approval manager: {'active' if orchestrator.approvals else 'inactive'}")
    print(f"  Database: {'connected' if orchestrator.database else 'offline'}")
    print(f"  MLOps: {len([c for c in caps if 'mlops' in c.tags or 'pipeline' in c.tags])} tracking capabilities")

    handler_class = type("Handler", (PlatformServerHandler,), {})
    handler_class.server_registry = registry
    handler_class.server_orchestrator = orchestrator
    handler_class.server_frontend_dir = FRONTEND_DIR

    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}"

    print(f"\n  Serving at: \033[1m{url}\033[0m")
    print(f"  Frontend:   {FRONTEND_DIR}")
    print("  Press Ctrl+C to stop\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
