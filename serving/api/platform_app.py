"""FastAPI application for the SilverWing Platform.

Consolidates the legacy ``ThreadingHTTPServer`` stack
(``sw_platform.api.PlatformHandler`` + the static-file handler from
``scripts/serve_platform.py``) onto FastAPI/uvicorn while preserving the exact
HTTP contract consumed by the frontend:

    GET  /health                     — health check (envelope)
    GET  /info                       — model/provider info (envelope)
    GET  /v1/capabilities            — registered capabilities (envelope)
    GET  /v1/gestures                — gesture → action mapping table
    GET  /v1/gestures/status         — subsystem availability + config
    GET  /v1/gestures/stats          — live system metrics (CPU, mem, battery)
    POST /generate                   — raw text generation (envelope)
    POST /v1/chat                    — orchestration loop (tool-use aware)
    POST /v1/chat/completions        — OpenAI-compatible chat completions
    POST /v1/tools/execute           — direct single tool execution

Route handlers are plain sync functions, so Starlette runs each request in
its threadpool — the same per-request concurrency as the old threading
server, without manual thread management or request queueing.

The legacy handler classes remain in ``sw_platform/api.py`` untouched for
backwards compatibility; new deployments should use this app via
``uvicorn serving.api.platform_app:create_app --factory`` or through
``scripts/serve_platform.py``.

Frontend rendering
------------------
When a :class:`FrontendController` is supplied the dashboard is **server-side
rendered** from Jinja2-style templates (``intelligence.webdev.templates``)
and static assets are fingerprinted via the ``AssetPipeline``.  The old
``SPAStaticFiles`` fallback is still used when only ``frontend_dir`` is
provided, preserving backward compatibility with existing deployments.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """POST /generate body."""

    prompt: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.0


class ChatRequestBody(BaseModel):
    """POST /v1/chat body."""

    message: str = ""
    max_rounds: int = 5


class ChatCompletionsMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionsRequest(BaseModel):
    """POST /v1/chat/completions body (OpenAI-compatible subset)."""

    messages: list[ChatCompletionsMessage] = Field(default_factory=list)
    model: str | None = None
    stream: bool = False
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


class AgenticRunRequest(BaseModel):
    """POST /v1/agentic/run body."""

    level: int | str = 1
    message: str = ""
    session_id: str = ""
    top_p: float = 0.9


class ToolExecuteRequest(BaseModel):
    """POST /v1/tools/execute body."""

    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api(success: bool, data: Any = None, error: str = "") -> dict[str, Any]:
    """Standard API response envelope (matches legacy ApiResponse.to_json)."""
    return {"success": success, "data": data, "error": error}


# ---------------------------------------------------------------------------
# Off-topic guard — the SFT model only knows math word problems
# ---------------------------------------------------------------------------

_MATH_HINT_RE = re.compile(
    r"\d|[\+\-\*/=^%]|\b("
    r"compute|calculate|solve|evaluate|simplify|percent|equation|algebra|"
    r"geometry|fraction|multiply|divide|add|subtract|sum|product|square|"
    r"root|average|mean|median|ratio|integer|decimal|variable)\b",
    re.IGNORECASE,
)

_OFF_TOPIC_REPLY = (
    "I'm Silverwing-v2-SFT - a small (102M) decoder trained only on math "
    "word problems, so I can't chat about general topics yet.\n\n"
    "Try asking me something like:\n"
    "  * What is 19% of 50?\n"
    "  * Solve for x: 3x + 7 = 22\n"
    "  * A train travels 120 km in 2 hours. What is its speed?"
)


def _off_topic_guard_enabled() -> bool:
    return os.environ.get("SILVERWING_OFF_TOPIC_GUARD", "1") == "1"


def _is_math_query(text: str) -> bool:
    return bool(_MATH_HINT_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Integrated agentic loop (M21): the model itself decides when to invoke
# capabilities. Tools are injected into its context as a text protocol; any
# <tool>{...}</tool> span it emits is executed by the platform and the result
# fed back until it answers. No named modules on the user side.
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = (
    "You are Silverwing. You can use tools inside your answer.\n"
    "To run a tool, write exactly:\n"
    '<tool>{"name": "...", "arguments": {...}}</tool>\n'
    "The platform replaces the span with <result>...</result> and you continue.\n"
    "Show any formula you use before computing. Answer the user directly when ready.\n"
    "Example:\n"
    "Area uses A = pi*r^2.\n"
    '<tool>{"name": "calculator", "arguments": {"expression": "3.14159*7**2"}}</tool>\n'
    "<result>153.93791</result>\n"
    "So the area is about 153.94 square units."
)

_TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)


def _extract_tool_call(text: str) -> tuple[str, dict] | None:
    match = _TOOL_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
        return str(payload.get("name", "")), payload.get("arguments") or {}
    except json.JSONDecodeError:
        return None


def _run_capability(registry: Any, name: str, arguments: dict) -> tuple[bool, str]:
    try:
        from intelligence.tools.protocol import ToolCall

        args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
        call = ToolCall(tool_name=name, arguments=args_str)
        result = registry.execute_call(call)
    except Exception as exc:
        return False, f"error: {exc}"
    if getattr(result, "success", True):
        return True, str(getattr(result, "output", "") or "")
    return False, f"error: {getattr(result, 'error', 'unknown')}"


def _messages_to_prompt(messages: list[ChatCompletionsMessage]) -> str:
    """Flatten an OpenAI-style message list into the SFT training format.

    Silverwing Decoder V2 was instruction-tuned on
    ``Question: {instruction}\\nAnswer: {response}`` pairs, so the
    conversation is flattened to that template (user turns become
    Questions, assistant turns become Answers).
    """
    parts: list[str] = []
    for msg in messages:
        content = msg.content
        if not content:
            continue
        if msg.role == "system":
            parts.append(content)
        elif msg.role == "assistant":
            parts.append(f"Answer: {content}")
        else:
            parts.append(f"Question: {content}")
    if parts and parts[-1].startswith("Question:"):
        # Trailing space matters: a bare "Answer:" tail makes the SFT model
        # emit EOS immediately; "Answer: " anchors continuation correctly.
        parts.append("Answer: ")
    return "\n".join(parts)


class SPAStaticFiles(StaticFiles):
    """Static files with SPA fallback — unknown paths serve index.html."""

    async def get_response(self, path: str, scope: Any) -> Any:
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            # Unknown route → SPA entry point
            response = await super().get_response("index.html", scope)
        if response.status_code == 404:
            response = await super().get_response("index.html", scope)
        return response


# ---------------------------------------------------------------------------
# Chat UI (self-contained page served at /chat)
# ---------------------------------------------------------------------------

CHAT_UI_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silverwing Chat</title>
<style>
  :root { --bg:#0b0e14; --panel:#121826; --edge:#1f2937; --ink:#e5edf8;
          --dim:#8fa3bd; --acc:#4da3ff; --ok:#39d98a; --bad:#ff6b6b;
          --user:#1c2a44; --bot:#161d2e; }
  * { box-sizing:border-box; margin:0; }
  html, body { height:100%; }
  body { background:var(--bg); color:var(--ink);
         font:15px/1.55 "Segoe UI",system-ui,sans-serif;
         display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:12px;
           padding:12px 20px; border-bottom:1px solid var(--edge);
           background:var(--panel); flex-wrap:wrap; }
  header h1 { font-size:16px; letter-spacing:.5px; }
  header h1 span { color:var(--acc); }
  #model-badge { font-size:12px; color:var(--dim);
                 border:1px solid var(--edge); border-radius:6px;
                 padding:2px 8px; }
  .spacer { flex:1; }
  header a { color:var(--acc); font-size:13px; text-decoration:none; }
  header nav a { color:var(--dim); padding:2px 4px; border-radius:5px; }
  header nav a:hover { color:var(--acc); background:var(--bot); }
  .params { display:flex; gap:14px; align-items:center;
            font-size:12.5px; color:var(--dim); flex-wrap:wrap; }
  .params label { display:flex; gap:5px; align-items:center; }
  input[type=number], input[type=text] {
      width:64px; background:#0d1320; color:var(--ink);
      border:1px solid var(--edge); border-radius:6px; padding:3px 6px;
      font:inherit; }
  #sysprompt { width:230px; }
  #clear { background:none; border:1px solid var(--edge); color:var(--bad);
           border-radius:6px; padding:3px 10px; cursor:pointer; font-size:12.5px; }
  #log { flex:1; overflow-y:auto; padding:20px; display:flex;
         flex-direction:column; gap:14px; max-width:860px;
         margin:0 auto; width:100%; }
  .msg { border-radius:12px; padding:10px 16px; max-width:82%;
         white-space:pre-wrap; word-break:break-word; }
  .user { align-self:flex-end; background:var(--user);
          border:1px solid #2a3b5e; }
  .bot  { align-self:flex-start; background:var(--bot);
          border:1px solid var(--edge); }
  .bot.error { border-color:var(--bad); color:var(--bad); }
  .meta { font-size:11.5px; color:var(--dim); margin-top:6px; }
  .cursor { display:inline-block; width:9px; height:17px;
            background:var(--acc); vertical-align:text-bottom;
            animation:blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  form { display:flex; gap:10px; padding:14px 20px 18px;
         border-top:1px solid var(--edge); background:var(--panel);
         max-width:900px; margin:0 auto; width:100%; }
  #input { flex:1; background:#0d1320; color:var(--ink);
           border:1px solid var(--edge); border-radius:10px;
           padding:11px 14px; font:inherit; outline:none; resize:none;
           min-height:46px; max-height:160px; }
  #input:focus { border-color:var(--acc); }
  button.send { background:var(--acc); border:0; color:#04101f;
                font-weight:700; padding:0 24px; border-radius:10px;
                cursor:pointer; font-size:15px; }
  button.send:disabled { opacity:.45; cursor:wait; }
</style>
</head>
<body>
<header>
  <h1>SILVERWING <span>// CHAT</span></h1>
  <span id="model-badge">loading…</span>
  <div class="spacer"></div>
  <nav class="params" style="gap:10px">
    <a href="/">command deck</a><a href="/chat">chat</a><a
      href="/agentic">agentic</a><a href="/workspace">workspace</a><a
      href="/docs">docs</a>
  </nav>
  <div class="params">
    <label>system <input type="text" id="sysprompt"
        placeholder="(optional)"></label>
    <label>temp <input type="number" id="temp" value="0.7"
        step="0.05" min="0" max="2"></label>
    <label>max&nbsp;tokens <input type="number" id="maxtok"
        value="128" step="16" min="8" max="512"></label>
    <button id="clear" type="button">clear chat</button>
    <a href="/agentic">agentic console →</a>
  </div>
</header>
<div id="log"></div>
<form id="composer">
  <textarea id="input" rows="1"
    placeholder="Message Silverwing…  (Enter to send, Shift+Enter for newline)"
    autofocus></textarea>
  <button class="send" id="send" type="submit">Send</button>
</form>
<script>
const $ = id => document.getElementById(id);
let history = [];   // [{role, content}]
let busy = false;

fetch('/info').then(r=>r.json()).then(j => {
  $('model-badge').textContent =
    j.success ? (j.data.model_id + ' · ' + j.data.provider)
              : 'no model';
});

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  $('log').appendChild(div);
  $('log').scrollTop = $('log').scrollHeight;
  return div;
}

function metaLine(div, txt) {
  let m = div.querySelector('.meta');
  if (!m) { m = document.createElement('div'); m.className='meta';
            div.appendChild(m); }
  m.textContent += txt;
}

async function send(text) {
  busy = true; $('send').disabled = true;
  addMsg('user', text);
  history.push({role:'user', content:text});
  const bot = addMsg('bot', '');
  const cur = document.createElement('span'); cur.className='cursor';
  bot.appendChild(cur);
  const t0 = performance.now();
  try {
    const body = {
      messages: ($('sysprompt').value.trim()
        ? [{role:'system', content:$('sysprompt').value.trim()}] : []
      ).concat(history),
      stream: true,
      temperature: parseFloat($('temp').value) || 0.7,
      max_tokens: parseInt($('maxtok').value) || 128,
    };
    const res = await fetch('/v1/chat/completions', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '', full = '', usage = null;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const parts = buf.split('\\n\\n'); buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data: ')) continue;
        const dataStr = line.slice(6);
        if (dataStr === '[DONE]') continue;
        try {
          const j = JSON.parse(dataStr);
          if (j.usage) usage = j.usage;
          const d = j.choices?.[0]?.delta?.content;
          if (d) {
            full += d;
            bot.textContent = full;
            bot.appendChild(cur);
            $('log').scrollTop = $('log').scrollHeight;
          }
        } catch (_) {}
      }
    }
    cur.remove();
    if (!full) bot.textContent = '(empty response - model emitted EOS)';
    history.push({role:'assistant', content:full});
    const secs = ((performance.now()-t0)/1000).toFixed(1);
    metaLine(bot, secs + 's' +
      (usage ? ' · ' + usage.completion_tokens + ' out · '
                  + usage.total_tokens + ' tok' : ''));
  } catch (e) {
    cur.remove();
    bot.textContent = 'request failed: ' + e.message;
    bot.classList.add('error');
  } finally { busy = false; $('send').disabled = false; $('input').focus(); }
}

$('composer').addEventListener('submit', e => {
  e.preventDefault();
  const text = $('input').value.trim();
  if (!text || busy) return;
  $('input').value = ''; $('input').style.height = 'auto';
  send(text);
});
$('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $('composer').requestSubmit();
  }
});
$('input').addEventListener('input', () => {
  $('input').style.height = 'auto';
  $('input').style.height =
    Math.min($('input').scrollHeight, 160) + 'px';
});
$('clear').onclick = () => { history = []; $('log').innerHTML = ''; };
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Agentic console UI (self-contained page served at /agentic)
# ---------------------------------------------------------------------------

AGENTIC_UI_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silverwing Agentic Console</title>
<style>
  :root { --bg:#0b0e14; --panel:#121826; --edge:#1f2937; --ink:#e5edf8;
          --dim:#8fa3bd; --acc:#4da3ff; --ok:#39d98a; --bad:#ff6b6b; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink);
         font:15px/1.5 "Segoe UI",system-ui,sans-serif; padding:24px; }
  .wrap { max-width:960px; margin:0 auto; }
  h1 { font-size:20px; letter-spacing:.4px; }
  h1 span { color:var(--acc); }
  .sub { color:var(--dim); font-size:13px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--edge);
          border-radius:12px; padding:16px; margin-bottom:14px; }
  label { display:block; font-size:12px; color:var(--dim);
          text-transform:uppercase; letter-spacing:.6px; margin-bottom:6px; }
  select, textarea { width:100%; background:#0d1320; color:var(--ink);
        border:1px solid var(--edge); border-radius:8px; padding:10px;
        font:inherit; outline:none; }
  select:focus, textarea:focus { border-color:var(--acc); }
  textarea { min-height:84px; resize:vertical; }
  button { background:var(--acc); border:0; color:#04101f; font-weight:700;
           padding:10px 22px; border-radius:8px; cursor:pointer;
           font-size:15px; margin-top:10px; }
  button:disabled { opacity:.5; cursor:wait; }
  #out { white-space:pre-wrap; }
  .lvl { color:var(--acc); font-weight:600; margin-bottom:8px; }
  .steps { margin-top:12px; border-top:1px dashed var(--edge); padding-top:10px; }
  .step { display:flex; gap:10px; padding:4px 0; font-size:13.5px; }
  .k { min-width:110px; color:var(--acc); font-family:ui-monospace,monospace; }
  .t { flex:1; word-break:break-word; }
  .meta { color:var(--dim); font-size:12.5px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Silverwing <span>Agentic Console</span></h1>
  <div class="sub" style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <span>six capability levels &middot; every run returns a full trace</span>
    <nav style="margin-left:auto">
      <a href="/" style="color:var(--dim);text-decoration:none;font-size:12.5px">command deck</a> ·
      <a href="/chat" style="color:var(--dim);text-decoration:none;font-size:12.5px">chat</a> ·
      <a href="/workspace" style="color:var(--dim);text-decoration:none;font-size:12.5px">workspace</a> ·
      <a href="/docs" style="color:var(--dim);text-decoration:none;font-size:12.5px">docs</a>
    </nav>
  </div>

  <div class="card">
    <label for="lvl">Capability level</label>
    <select id="lvl"></select>
    <label for="msg" style="margin-top:12px">Message / goal</label>
    <textarea id="msg" placeholder="Ask anything, or give an autonomous goal..."></textarea>
    <button id="go">Run</button>
    <span id="stat" class="meta"></span>
  </div>

  <div class="card" id="res" style="display:none">
    <div class="lvl" id="rlvl"></div>
    <div id="out"></div>
    <div class="meta" id="rmeta"></div>
    <div class="steps" id="steps"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
fetch('/v1/agentic/levels').then(r=>r.json()).then(j=>{
  if(!j.success) return;
  $('lvl').innerHTML = j.data.map(l =>
    `<option value="${l.level}">${l.label}</option>`).join('');
});
$('go').onclick = async () => {
  const message = $('msg').value.trim();
  if (!message) return;
  $('go').disabled = true; $('stat').textContent = 'running…';
  try {
    const r = await fetch('/v1/agentic/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({level:+$('lvl').value, message})
    });
    const j = await r.json();
    if (!j.success) { $('stat').textContent = 'error: '+j.error; return; }
    const d = j.data;
    $('res').style.display = 'block';
    $('rlvl').textContent = d.level_label + (d.success ? '' : '  · did not converge');
    $('out').textContent = d.final_text || '(no text)';
    $('rmeta').textContent =
      d.elapsed_seconds + 's · ' + d.steps.length + ' steps';
    $('steps').innerHTML = d.steps.map(s =>
      `<div class="step"><span class="k">${s.kind}</span>` +
      `<span class="t">${(s.detail||'').replace(/</g,'&lt;')}</span></div>`).join('');
    $('stat').textContent = '';
  } catch (e) { $('stat').textContent = 'request failed: ' + e; }
  finally { $('go').disabled = false; }
};
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    registry: Any,
    orchestrator: Any,
    frontend_dir: Path | None = None,
    frontend_controller: Any = None,
) -> FastAPI:
    """Build the platform FastAPI app around live platform objects.

    Parameters
    ----------
    registry
        Capability registry instance (or ``None`` for a UI-only shell).
    orchestrator
        Orchestration layer instance (or ``None``).
    frontend_dir
        Legacy static-HTML directory for backward compatibility.  Ignored
        when *frontend_controller* is supplied.
    frontend_controller
        Optional :class:`~silverwing_platform.frontend.FrontendController`
        that renders the dashboard from templates and serves static assets
        via the project's own ``intelligence.webdev`` framework.
    """
    app = FastAPI(
        title="SilverWing Platform",
        description="Capability registry + chat orchestration + tool execution",
        version="1.0.0",
    )
    app.state.registry = registry
    app.state.orchestrator = orchestrator
    app.state.frontend_dir = frontend_dir
    app.state.frontend_controller = frontend_controller

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(_api(success=True, data={"status": "ok"}))

    @app.get("/info")
    def info() -> JSONResponse:
        generator = getattr(orchestrator, "generator", None)
        if generator is None:
            return JSONResponse(
                _api(success=False, error="No model provider loaded"), status_code=503
            )
        return JSONResponse(_api(
            success=True,
            data={
                "model_id": getattr(generator, "model_id", type(generator).__name__),
                "provider": type(generator).__name__,
            },
        ))

    # ------------------------------------------------------------------
    # Capabilities / tools
    # ------------------------------------------------------------------

    @app.get("/v1/capabilities")
    def capabilities() -> JSONResponse:
        if registry is None:
            return JSONResponse(
                _api(success=False, error="No capability registry"), status_code=503
            )
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
            for cap in registry.list(enabled_only=False)
        ]
        return JSONResponse(_api(success=True, data=caps))

    @app.post("/v1/tools/execute")
    def tools_execute(body: ToolExecuteRequest) -> JSONResponse:
        if not body.tool:
            return JSONResponse(
                _api(success=False, error="Missing 'tool'"), status_code=400
            )
        if registry is None:
            return JSONResponse(
                _api(success=False, error="No capability registry"), status_code=503
            )
        try:
            from intelligence.tools.protocol import ToolCall

            call = ToolCall(
                tool_name=body.tool,
                arguments=",".join(f"{k}={v}" for k, v in body.arguments.items()),
            )
            result = registry.execute_call(call)
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(
            success=True,
            data={
                "tool": result.tool_name,
                "output": result.output,
                "success": result.success,
                "error": result.error,
            },
        ))

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @app.post("/v1/chat")
    def chat(body: ChatRequestBody) -> JSONResponse:
        if not body.message:
            return JSONResponse(
                _api(success=False, error="Missing 'message'"), status_code=400
            )
        if orchestrator is None:
            return JSONResponse(
                _api(success=False, error="No orchestrator loaded"), status_code=503
            )
        try:
            from sw_platform.orchestration.orchestrator import ChatRequest

            response = orchestrator.handle_request(
                ChatRequest(message=body.message, max_rounds=body.max_rounds)
            )
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(success=True, data=response.to_dict()))

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionsRequest) -> Any:
        """OpenAI-compatible endpoint backed directly by the model provider."""
        if not body.messages:
            return JSONResponse(
                _api(success=False, error="'messages' must be a non-empty list"),
                status_code=400,
            )
        generator = getattr(orchestrator, "generator", None)
        if generator is None:
            return JSONResponse(
                _api(success=False, error="No model provider loaded"), status_code=503
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = (
            body.model
            or getattr(generator, "model_id", None)
            or "silverwing-v2"
        )

        # Off-topic guard: the SFT checkpoint only knows math word problems,
        # so non-math input would produce incoherent math-scaffold text.
        last_user = next(
            (
                m for m in reversed(body.messages)
                if m.role == "user" and m.content.strip()
            ),
            None,
        )
        if _off_topic_guard_enabled() and last_user and not _is_math_query(last_user.content):
            reply_text = _OFF_TOPIC_REPLY
            usage_block = {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }
            tool_events: list[dict[str, Any]] = []
        else:
            try:
                from silverwing_platform.models import GenerationConfig, InferenceRequest

                # Integrated agentic loop: model decides when to use tools.
                messages_for_model = [
                    m.model_dump() if hasattr(m, "model_dump") else dict(m)
                    for m in body.messages
                ]
                if not any(
                    (m.get("role") if isinstance(m, dict) else m.role) == "system"
                    for m in messages_for_model
                ):
                    messages_for_model.insert(0, {"role": "system", "content": _AGENT_SYSTEM})

                # A trailing assistant message acts as a prefill: its content
                # counts as part of the model output for tool extraction.
                prefill = ""
                if messages_for_model:
                    last = messages_for_model[-1]
                    last_role = last.get("role") if isinstance(last, dict) else last.role
                    last_content = last.get("content") if isinstance(last, dict) else last.content
                    if last_role == "assistant" and last_content:
                        prefill = str(last_content)

                tool_events = []
                response = None
                max_rounds = 3
                for round_idx in range(max_rounds + 1):
                    prompt = _messages_to_prompt([
                        ChatCompletionsMessage(**m) if isinstance(m, dict) else m
                        for m in messages_for_model
                        if (m.get("content") if isinstance(m, dict) else m.content)
                    ])
                    response = generator.infer(
                        InferenceRequest(
                            prompt=prompt,
                            config=GenerationConfig(
                                max_new_tokens=body.max_tokens,
                                temperature=body.temperature,
                                top_p=body.top_p,
                            ),
                        )
                    )
                    combined = prefill + (response.text or "")
                    call = _extract_tool_call(combined)
                    if call is None or round_idx == max_rounds:
                        reply_text = combined
                        break
                    name, arguments = call
                    ok, output = _run_capability(registry, name, arguments)
                    tool_events.append({
                        "name": name,
                        "arguments": arguments,
                        "ok": ok,
                        "output": output[:300],
                    })
                    messages_for_model.append({"role": "assistant", "content": combined})
                    messages_for_model.append({
                        "role": "user",
                        "content": f"<result>{output}</result>\nContinue your answer to the user.",
                    })
                    prefill = ""

                usage = getattr(response, "usage", None) or {}
            except Exception as exc:
                return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
            usage_block = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("generated_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        if body.stream:
            # Simulated token streaming: generation already completed (CPU
            # batch decode), so the text is emitted as OpenAI-style SSE
            # chunks in small word groups for progressive client rendering.
            created = int(time.time())

            def sse() -> Iterator[str]:
                def chunk(delta: dict[str, Any], finish: str | None) -> str:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                yield f"data: {json.dumps({'type': 'tool_events', 'tools': tool_events})}\n\n"
                yield chunk({"role": "assistant"}, None)
                words = reply_text.split(" ")
                for i in range(0, len(words), 3):
                    group = " ".join(words[i : i + 3])
                    if i + 3 < len(words):
                        group += " "
                    yield chunk({"content": group}, None)
                    time.sleep(0.03)
                yield chunk({}, "stop")
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [],
                    "usage": usage_block,
                }
                yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"

            return StreamingResponse(
                sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_block,
            "tool_calls": tool_events,
        }
        return JSONResponse(payload)

    @app.post("/generate")
    def generate(body: GenerateRequest) -> JSONResponse:
        """Raw text generation against the platform model provider."""
        if not body.prompt:
            return JSONResponse(
                _api(success=False, error="Missing 'prompt'"), status_code=400
            )
        generator = getattr(orchestrator, "generator", None)
        if generator is None:
            return JSONResponse(
                _api(success=False, error="No model provider loaded"), status_code=503
            )
        try:
            from silverwing_platform.models import GenerationConfig, InferenceRequest

            response = generator.infer(InferenceRequest(
                prompt=body.prompt,
                config=GenerationConfig(
                    max_new_tokens=body.max_new_tokens,
                    temperature=max(body.temperature, 1e-8),
                ),
            ))
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(
            success=True,
            data={"text": response.text, "prompt": body.prompt},
        ))

    # ------------------------------------------------------------------
    # Gesture OS endpoints
    # ------------------------------------------------------------------

    @app.get("/v1/gestures")
    def gestures() -> JSONResponse:
        try:
            from sw_platform.tools.gesture import get_gesture_registry

            data = get_gesture_registry()
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(success=True, data={"gestures": data["gestures"]}))

    @app.get("/v1/gestures/status")
    def gestures_status() -> JSONResponse:
        try:
            from sw_platform.tools.gesture import get_gesture_registry

            data = get_gesture_registry()
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(success=True, data=data["status"]))

    @app.get("/v1/gestures/stats")
    def gestures_stats() -> JSONResponse:
        try:
            from sw_platform.tools.gesture import GestureCapabilityProvider

            stats = GestureCapabilityProvider().get_system_stats()
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(success=True, data=stats))

    # ------------------------------------------------------------------
    # Agentic AI — six capability levels (L1..L6)
    # ------------------------------------------------------------------

    engine_holder: dict[str, Any] = {}

    def _get_engine() -> Any:
        if "engine" not in engine_holder:
            from foundation.agentic.engine import AgenticEngine

            engine_holder["engine"] = AgenticEngine(allowed_paths=[str(Path.cwd())])
        return engine_holder["engine"]

    @app.get("/v1/agentic/levels")
    def agentic_levels() -> JSONResponse:
        from foundation.agentic.levels import AgentLevel

        data = [{"level": int(lv), "label": lv.label} for lv in AgentLevel]
        return JSONResponse(_api(success=True, data=data))

    @app.get("/v1/agentic/tools")
    def agentic_tools() -> JSONResponse:
        try:
            tools = _get_engine().tool_catalog()
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(success=True, data={"tools": tools}))

    @app.post("/v1/agentic/run")
    def agentic_run(body: AgenticRunRequest) -> JSONResponse:
        if not body.message:
            return JSONResponse(
                _api(success=False, error="Missing 'message'"), status_code=400
            )
        try:
            trace = _get_engine().run(
                body.level, body.message, session_id=body.session_id
            )
        except ValueError as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=400)
        except Exception as exc:
            return JSONResponse(_api(success=False, error=str(exc)), status_code=500)
        return JSONResponse(_api(success=True, data=trace.to_dict()))

    @app.get("/agentic", response_class=HTMLResponse)
    def agentic_ui() -> str:
        return AGENTIC_UI_PAGE

    @app.get("/chat", response_class=HTMLResponse)
    def chat_ui() -> str:
        return CHAT_UI_PAGE

    # ------------------------------------------------------------------
    # Frontend (mounted last so API routes take precedence)
    # ------------------------------------------------------------------

    if frontend_controller is not None:
        # ── Modern: server-side rendered dashboard via FrontendController ──

        from silverwing_platform.frontend import FrontendController
        from silverwing_platform.frontend.controller import PlatformContext

        def _build_context() -> PlatformContext:
            ctx = PlatformContext()
            gen = getattr(orchestrator, "generator", None)
            if gen is not None:
                ctx.model_name = getattr(gen, "_model_id", type(gen).__name__.upper())
            if registry is not None:
                caps = registry.list(enabled_only=True)
                ctx.capabilities = FrontendController.caps_to_summaries(caps)
                ctx.capability_count = len(caps)
                ctx.cap_json = FrontendController.caps_to_json(caps)
            return ctx

        @app.get("/", response_class=HTMLResponse)
        def serve_dashboard() -> str:
            return frontend_controller.render_dashboard(_build_context())

        # Serve fingerprinted / raw static assets
        static_path = frontend_controller.static_dir
        if static_path.is_dir():
            app.mount(
                "/static",
                StaticFiles(directory=str(static_path)),
                name="static",
            )

        # SPA fallback for client-side routes
        @app.get("/{path:path}", response_class=HTMLResponse)
        def spa_fallback(request: Request) -> str:
            raw = request.url.path.lstrip("/")
            if raw.startswith("_"):
                return ""
            return frontend_controller.render_dashboard(_build_context())

    elif frontend_dir is not None and Path(frontend_dir).is_dir():
        # ── Legacy: static HTML SPA fallback ──
        app.mount(
            "/",
            SPAStaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )

    # ------------------------------------------------------------------
    # Workspace SPA (agent harness UI) served at /workspace
    # ------------------------------------------------------------------

    workspace_dir = Path(__file__).resolve().parents[1] / "frontend" / "static"
    if frontend_controller is None and workspace_dir.is_dir():
        app.mount(
            "/workspace",
            SPAStaticFiles(directory=str(workspace_dir), html=True),
            name="workspace",
        )

    # ------------------------------------------------------------------
    # WebSocket agent bridge — powers the /workspace UI
    # ------------------------------------------------------------------

    ws_sessions: dict[str, Any] = {}

    def _get_harness(session_id: str) -> Any:
        """Return a cached PydanticAgentHarness, creating one on first use.

        Raises ImportError when pydantic-ai is not installed.
        """
        if session_id in ws_sessions:
            return ws_sessions[session_id]
        from sw_platform.harness.agent import HarnessConfig, PydanticAgentHarness

        harness = PydanticAgentHarness(HarnessConfig(
            model=os.environ.get("SILVERWING_AGENT_MODEL", "openai:silverwing-v2"),
            permission_level=os.environ.get("SILVERWING_AGENT_PERMISSION", "read"),
            read_only_mode=True,
        ))
        ws_sessions[session_id] = harness
        return harness

    @app.websocket("/ws/{session_id}")
    async def ws_agent_bridge(ws: WebSocket, session_id: str) -> None:
        """Agent protocol consumed by serving/frontend/static/workspace.js:

        client → {"action": "message|list_tools|ping|reset", "content": ...}
        server → {"type": "session_ready|typing|response|tool_call|audit|
                  tools|error|pong|reset_done", ...}

        Stays connected even without pydantic-ai so the UI degrades
        gracefully (errors surfaced as events instead of reconnect loops).
        """
        import logging

        logger = logging.getLogger("serving.ws")
        await ws.accept()

        try:
            harness = _get_harness(session_id)
        except ImportError as exc:
            harness = None
            await ws.send_json({
                "type": "error",
                "error": f"Agent harness unavailable ({exc})",
            })
        except Exception as exc:
            harness = None
            await ws.send_json({"type": "error", "error": str(exc)})

        if harness is not None:
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
                if action == "ping":
                    await ws.send_json({"type": "pong"})
                    continue
                if harness is None:
                    await ws.send_json({
                        "type": "error",
                        "error": "Agent harness unavailable - pip install pydantic-ai",
                    })
                    continue

                if action == "message":
                    content = data.get("content", "")
                    await ws.send_json({"type": "typing", "status": "started"})
                    try:
                        import asyncio

                        response = await asyncio.to_thread(harness.run, content)
                        await ws.send_json({
                            "type": "response",
                            "text": response.text,
                            "elapsed_seconds": response.elapsed_seconds,
                            "success": response.success,
                        })
                        for tc in response.tool_calls:
                            await ws.send_json({
                                "type": "tool_call",
                                "data": tc.to_dict(),
                            })
                        for event in harness.audit_log[-10:]:
                            await ws.send_json({"type": "audit", "data": event})
                    except Exception as exc:
                        logger.exception("harness run failed")
                        await ws.send_json({"type": "error", "error": str(exc)})
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

                elif action == "reset":
                    harness._conversation_history.clear()
                    await ws.send_json({"type": "reset_done"})

        except WebSocketDisconnect:
            ws_sessions.pop(session_id, None)

    from serving.gateway.middleware import install_gateway

    install_gateway(app)
    return app
