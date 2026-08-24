"""Serve a Silverwing checkpoint with the full interactive platform.

One command = trained model + orchestration + web dashboard::

    python scripts/serve_model.py                       # configs/serving_production.yaml
    python scripts/serve_model.py --port 9000           # override port
    python scripts/serve_model.py --checkpoint PATH     # ad-hoc checkpoint

Interactive surfaces (http://127.0.0.1:8000):
    /                tactical dashboard - chat, capabilities, tools
    /agentic         agentic console (capability levels L1-L6)
    /docs            OpenAPI explorer

API endpoints:
    POST /v1/chat                   orchestration loop (tool-use aware)
    POST /v1/chat/completions       OpenAI-compatible chat completions
    POST /v1/tools/execute          direct tool execution
    POST /generate                  raw text generation
    GET  /health | /info | /v1/capabilities | /v1/agentic/*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from serving.runtime.runtime import Runtime, RuntimeConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a Silverwing model")
    parser.add_argument("--config", default="configs/serving_production.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    if cfg_path.exists():
        config = RuntimeConfig.from_yaml(str(cfg_path))
    else:
        print(f"[warn] config {cfg_path} not found - using defaults")
        config = RuntimeConfig()
    config.host = args.host or "127.0.0.1"
    if args.port:
        config.port = args.port
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.tokenizer_dir:
        config.tokenizer_dir = args.tokenizer_dir

    # Load model + build orchestrator/capability registry (intelligence mode)
    runtime = Runtime(config, log=print, intelligence_enabled=True)
    runtime.load()

    # Self-hosted agent harness: pydantic_ai talks to OUR completions API,
    # so the /workspace WebSocket UI runs fully offline against the
    # locally served Silverwing model.
    import os

    base_url = f"http://127.0.0.1:{config.port}/v1"
    os.environ.setdefault("SILVERWING_AGENT_MODEL", "openai:silverwing-v2")
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_KEY", "local")

    from serving.api.platform_app import create_app  # noqa: E402

    frontend_controller = None
    try:
        from silverwing_platform.frontend import FrontendController  # noqa: E402

        frontend_controller = FrontendController()
    except Exception as exc:
        print(f"[warn] dashboard unavailable ({exc}) - API-only mode")

    app = create_app(
        registry=runtime.orchestrator.registry,
        orchestrator=runtime.orchestrator,
        frontend_controller=frontend_controller,
    )

    import uvicorn  # noqa: E402

    print(f"Dashboard: http://{config.host}:{config.port}/")
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
