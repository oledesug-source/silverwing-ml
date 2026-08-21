# SilverWing Runtime v1 — Controlled Intelligence Platform

The Controlled Intelligence Platform (`sw_platform/`) is the orchestration
layer that connects `foundation/` (model, training, serving) and
`intelligence/` (planner, reasoner, memory, tools) into a production
agent system. The LLM **proposes** — the platform **decides** and
**executes**.

Every capability is schema-defined, permission-gated (L0–L5),
resource-bounded (sandbox), and audit-logged.

## Package layout

```
intelligence/
├── transformers/                     # Steps 01-02: Attention, Transformer, BPE, positional encoding (34 tests)
│   ├── __init__.py
│   ├── attention.py
│   ├── model.py
│   ├── positional_encoding.py
│   └── bpe_tokenizer.py
├── embeddings/                       # Steps 05-06: TF-IDF, word/sentence embeddings, hashing (28 tests)
│   └── __init__.py
├── vector_db/                        # Steps 05-06: VectorStore, HybridSearch, VectorIndex (25 tests)
│   └── __init__.py
├── mcp/                              # Steps 07-09: MCPServer, MCPClient, tools/resources/prompts (16 tests)
│   └── __init__.py
├── peft/                             # Steps 10-11: LoRA, LoRAAdapter, LoRATrainer (17 tests)
│   └── __init__.py
├── prompt/                           # Step 03: Prompt templates, CoT, structured output (12 tests)
│   └── __init__.py
├── rag/                              # Steps 05-06: Chunker, Retriever, RAGPromptBuilder, RAGPipeline (15 tests)
│   └── __init__.py
├── multimodal/                       # Steps 10-11: Image/Audio encoders (17 tests)
│   └── __init__.py
└── observability/                    # Steps 14-16: Tracing, metrics, guardrails, red team (18 tests)
    └── __init__.py
sw_platform/
├── __init__.py                       # Public API (15 exports)
├── api.py                            # PlatformHandler (HTTP endpoints)
├── capabilities/
│   ├── schema.py                     # CapabilitySchema (id, version, permissions, risk)
│   ├── registry.py                   # CapabilityRegistry (wraps ToolRegistry)
│   └── discovery.py                  # CapabilityDiscovery (task-aware selection)
├── context/
│   ├── models.py                     # SessionState + RequestContext
│   └── builder.py                    # ContextBuilder (factory methods)
├── permissions/
│   └── policy.py                     # PermissionLevel L0–L5 + PermissionEvaluator
├── sandbox/
│   └── executor.py                   # ResourceLimits + SandboxExecutor
├── audit/
│   └── events.py                     # AuditEvent + AuditLog (ring buffer)
├── orchestration/
│   ├── execution_loop.py             # ExecutionLoop (bounded propose-validate-execute)
│   └── orchestrator.py               # Orchestrator + ChatRequest/ChatResponse
├── harness/                          # Layer 2: Agent harness (pydantic_ai integration)
│   ├── __init__.py
│   ├── core.py                       # ExecutionResult, ToolSpec, ToolProvider (Protocol)
│   └── agent.py                      # PydanticAgentHarness, HarnessConfig, AgentResponse, create_harness_agent
├── tools/                            # Layer 3: Python automation tooling
│   ├── __init__.py
│   ├── code_execution.py             # CodeExecutionProvider (run_python, python_ast)
│   ├── database.py                   # DatabaseProvider (sql_query, sql_list_tables, sql_schema, sql_explain)
│   ├── filesystem.py                 # FilesystemProvider (read_file, write_file, list_directory, move_file, delete_file)
│   ├── git.py                        # GitProvider (git_status, git_diff, git_log, git_add, git_commit, git_blame, git_clone)
│   └── web_automation.py             # WebAutomationProvider (web_fetch, web_scrape, web_form_fill)
└── coder/                            # Layer 4: Software & coding capabilities
    ├── __init__.py
    ├── core.py                       # CoderProvider, RepoContext, DockerSandbox, StructuredOutput
    └── models.py                     # CodePatch, CodeExplanation (pydantic models for structured output)
serving/
├── api/
│   ├── server.py                     # Legacy inference runtime API
│   └── fastapi_bridge.py             # Layer 2 Bridge: WebSocket + SSE → agent harness
└── frontend/static/                  # Layer 1: Enterprise workspace UI
    ├── index.html                    # Workspace layout (collapsible sidebars, status chips, multi-terminal)
    ├── design-system.css             # Cloudscape/Ant/Fluent-inspired enterprise design tokens
    ├── workspace.css                 # Workspace-specific styling
    └── workspace.js                  # UI interactions + WebSocket client
```

## Core principles

1. **LLM proposes, runtime decides** — model outputs are parsed for tool
   calls but never executed directly
2. **Schema-based capabilities** — every capability carries version,
   input/output schemas, permission requirements, risk level, and timeout
3. **Permission levels L0–L5** — read-only (L0) through full system (L5)
4. **Resource-bounded sandbox** — timeout, path restrictions, file size limits
5. **Structured audit trail** — every action recorded as an `AuditEvent`
6. **Stateless REST API** — POST /v1/chat, POST /v1/tools/execute,
   GET /v1/capabilities
7. **Bounded execution loop** — configurable max steps, fallback to
   heuristic engines when no generator is available

## Core abstractions

### CapabilitySchema (`capabilities/schema.py`)

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Auto-generated unique ID |
| `name` | `str` | Unique name |
| `version` | `str` | Semver (default `"1.0.0"`) |
| `description` | `str` | Human-readable description |
| `input_schema` | `dict` | JSON-Schema for inputs |
| `output_schema` | `dict` | JSON-Schema for outputs |
| `permissions_required` | `list[str]` | Min permission level(s) |
| `risk_level` | `str` | `"low"` / `"medium"` / `"high"` / `"critical"` |
| `timeout_seconds` | `float` | Max execution time |
| `execution_mode` | `str` | `"sync"` or `"async"` |
| `enabled` | `bool` | Whether currently available |
| `capability_type` | `str` | `"tool"` / `"reasoning"` / `"generation"` |
| `fn` | `Callable \| None` | Implementation function |
| `tags` | `list[str]` | Discovery tags |
| `source` | `str` | `"builtin"` / `"user"` / `"external"` |

### CapabilityRegistry (`capabilities/registry.py`)

Wraps `intelligence.tools.protocol.ToolRegistry` (adapter pattern).

- `register(schema)` — adds to both internal dict and ToolRegistry
- `unregister(name)` — removes a capability
- `get(name)` / `list(enabled_only)` / `search(query, tags, type)`
- `enable(name)` / `disable(name)` — toggle availability
- `system_prompt()` / `parse_calls(text)` / `execute_call(call)`
- `format_results(results)` / `to_tool_registry()`

### CapabilityDiscovery (`capabilities/discovery.py`)

Task-aware capability selection using tag matching and keyword scoring.
No AI dependency — pure heuristic.

```python
discovery = CapabilityDiscovery(registry)
relevant = discovery.find_for_task("calculate 2 + 2", context)
```

### SessionState + RequestContext (`context/models.py`)

```
SessionState
  ├── session_id
  ├── user_id
  ├── working_memory (WorkingMemory)
  └── metadata

RequestContext
  ├── request_id
  ├── session → SessionState
  ├── user_message
  ├── capabilities_used: list[str]
  ├── tool_results: list[ToolResult]
  ├── max_rounds: int
  └── metadata: dict
```

Helper methods: `add_user_message()`, `add_tool_result(result)`,
`add_assistant_message(text)`.

### ContextBuilder (`context/builder.py`)

- `ContextBuilder.from_request(message, max_rounds, ...)` — factory
- `ContextBuilder.build_system_prompt(registry, permission_level)` —
  generates prompt section filtering by permission level

### PermissionLevel L0–L5 (`permissions/policy.py`)

| Level | Name | Capabilities |
|---|---|---|
| L0 | `read` | calculator, read_file |
| L1 | `write` | File writes |
| L2 | `execute` | Code execution |
| L3 | `network` | Network access |
| L4 | `admin` | System administration |
| L5 | `system` | Full system access |

`PermissionPolicy(level, allowed_tools, denied_tools, require_sandbox)`
defines boundaries. `PermissionEvaluator.is_allowed(cap)` returns
`(bool, reason)`. High/critical risk capabilities auto-require sandbox.

### SandboxExecutor (`sandbox/executor.py`)

```
ResourceLimits
  ├── max_memory_bytes: int | None
  ├── max_file_size_bytes: int | None
  ├── max_execution_time: float (default 30s)
  ├── allowed_paths: list[str]
  ├── blocked_paths: list[str]
  └── network_allowed: bool
```

- `check_path(path)` — validates against allow/block lists
- `check_file_size(bytes)` — validates against limits
- `execute(fn, cap_id, **kwargs)` — timeout via threading + error boundary

### AuditEvent + AuditLog (`audit/events.py`)

```
AuditEvent
  ├── event_id, timestamp
  ├── request_id, session_id
  ├── action: str ("tool_call", "permission_denied", etc.)
  ├── capability_id, status, detail
  ├── elapsed_ms
  └── metadata: dict
```

`AuditLog` — in-memory ring buffer with `record()`, `query()`, `recent()`,
`clear()`. Max 10,000 entries (configurable). Events are dict-serializable.

### ExecutionLoop (`orchestration/execution_loop.py`)

Bounded propose-validate-execute cycle:

```
for step in range(max_steps):
  1. Parse tool calls from model output
  2. If none → return (done)
  3. For each call:
     a. Permission check
     b. Sandbox execute
     c. Audit log
  4. Format results → feed back to model
```

### Orchestrator (`orchestration/orchestrator.py`)

The main entry point. Combines registry, generator, permissions, sandbox,
and audit into a single `handle_request(ChatRequest) → ChatResponse`.

```python
orch = Orchestrator(registry=registry, generator=gen)
response = orch.handle_request(ChatRequest(message="What is 2+2?"))
print(response.text, response.rounds, response.audit_events)
```

`ChatResponse.to_dict()` serializes for JSON transport. Includes full
audit trail.

### PlatformHandler (`api.py`)

Extends `SilverwingHandler` with versioned endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat` | Full orchestration loop |
| `POST` | `/v1/tools/execute` | Direct tool execution |
| `GET` | `/v1/capabilities` | List registered capabilities |
| `POST` | `/generate` | Raw generation (legacy) |
| `GET` | `/health` | Health check |
| `GET` | `/info` | Model info |

## Dependency flow

```
sw_platform/api.py
  → sw_platform/orchestration/orchestrator.py
      → sw_platform/orchestration/execution_loop.py
          → sw_platform/capabilities/registry.py
              → intelligence/tools/protocol.py
          → sw_platform/permissions/policy.py
          → sw_platform/sandbox/executor.py
          → sw_platform/audit/events.py
      → sw_platform/context/builder.py
          → sw_platform/context/models.py
              → intelligence/memory/context.py
      → serving/api/server.py
```

No torch dependency at import time. All torch-dependent modules are
imported lazily. The entire package is testable without torch.

### Layer 2 bridge dependency flow

```
serving/api/fastapi_bridge.py
  → sw_platform/harness/agent.py (PydanticAgentHarness)
      → sw_platform/harness/core.py (ExecutionResult, ToolSpec)
      → sw_platform/tools/ (CodeExecution, Database, Filesystem, Git, WebAutomation)
      → sw_platform/coder/core.py (CoderProvider, RepoContext, DockerSandbox, StructuredOutput)
      → sw_platform/coder/models.py (CodePatch, CodeExplanation)
      → pydantic_ai.Agent (LLM integration)
```

### Layer 2 Bridge — FastAPI WebSocket (`serving/api/fastapi_bridge.py`)

Exposes the agent harness via WebSocket and SSE using FastAPI.  Streams
terminal logs, UI states, and responses to the frontend in real time.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend `index.html` |
| `/health` | GET | Health check |
| `/tools` | GET | Lists all available tools |
| `/chat` | POST | Synchronous chat (creates a session, runs agent, returns `ChatResponse`) |
| `/ws/{session_id}` | WebSocket | Real-time streaming — sends `session_ready`, `typing`, `response`, `tool_call`, `audit`, `error` messages |
| `/sessions/{session_id}/tools` | GET | Lists tools for a specific session |

**WebSocket protocol:**

```
Client → Server: {"action": "message", "content": "..."}
Server → Client: {"type": "session_ready", "session_id": "...", "tools": [...]}
Server → Client: {"type": "response", "text": "...", "success": true}
Server → Client: {"type": "tool_call", "data": {...}}
Server → Client: {"type": "audit", "data": {...}}
Server → Client: {"type": "typing", "status": "ended"}
```

Run the bridge:

```bash
uvicorn serving.api.fastapi_bridge:app --reload --host 0.0.0.0 --port 8080
```

### Layer 2 — Agent Harness (`sw_platform/harness/`)

The harness is the model's operational infrastructure.  Built on
[pydantic_ai](https://github.com/pydantic/pydantic-ai), it manages
long-term memory, context windows, error handling, and self-correction.

| Symbol | Source | Description |
|---|---|---|
| `PydanticAgentHarness` | `harness/agent.py` | Main agent class — wraps pydantic_ai `Agent`, registers tools, runs the propose-execute loop |
| `HarnessConfig` | `harness/agent.py` | Configuration: `model`, `max_rounds`, `system_prompt`, `permission_level` |
| `AgentResponse` | `harness/agent.py` | Structured response: `text`, `tool_calls`, `success`, `error`, `audit_log` |
| `ToolCallRecord` | `harness/agent.py` | Record of a tool call: `name`, `arguments`, `result`, `elapsed_seconds` |
| `create_harness_agent(model, ...)` | `harness/agent.py` | Factory function for the pydantic_ai `Agent` with tools bound |
| `ExecutionResult` | `harness/core.py` | Dataclass: standardized result from any tool (`tool_name`, `success`, `output`, `error`, `elapsed_seconds`, `metadata`) |
| `ToolSpec` | `harness/core.py` | Schema for a registered tool (`name`, `description`, `parameters`, `risk_level`, `permission_required`, `tags`) |
| `ToolProvider` | `harness/core.py` | `Protocol` defining the interface for all tool providers (`get_tools()` + `execute()`) |

### Layer 3 — Python Automation & Framework Tooling (`sw_platform/tools/`)

System execution, web automation, database, filesystem, and git tools
that give the agent deep system integration.

| Provider | Source | Tools |
|---|---|---|
| `CodeExecutionProvider` | `tools/code_execution.py` | `run_python` (subprocess), `python_ast` (safe eval) |
| `DatabaseProvider` | `tools/database.py` | `sql_query`, `sql_list_tables`, `sql_schema`, `sql_explain` (SQLite) |
| `FilesystemProvider` | `tools/filesystem.py` | `read_file`, `write_file`, `list_directory`, `move_file`, `delete_file` (path-allowlisted) |
| `GitProvider` | `tools/git.py` | `git_status`, `git_diff`, `git_log`, `git_add`, `git_commit`, `git_blame`, `git_clone` |
| `WebAutomationProvider` | `tools/web_automation.py` | `web_fetch`, `web_scrape`, `web_form_fill` (httpx fallback when Playwright not installed) |

Each provider implements the `ToolProvider` protocol: `get_tools()` returns
a list of `ToolSpec` objects and `execute(tool_name, **kwargs)` returns an
`ExecutionResult`.

### Layer 4 — Software & Coding Capabilities (`sw_platform/coder/`)

Code interpreter, repository contexting, sandboxed execution, and
structured output for the agent.

| Symbol | Source | Description |
|---|---|---|
| `CoderProvider` | `coder/core.py` | Provides coding tools: `explain_code`, `generate_patch`, `review_code` |
| `RepoContext` | `coder/core.py` | Git-aware repository context — file indexing, language detection, search, repo hashing |
| `DockerSandbox` | `coder/core.py` | Sandboxed code execution — uses Docker when available, falls back to local subprocess |
| `StructuredOutput` | `coder/core.py` | Validates LLM JSON output against pydantic models (`validate(data, model_cls)`) |
| `CodePatch` | `coder/models.py` | Pydantic model: `file_path`, `old_content`, `new_content` |
| `CodeExplanation` | `coder/models.py` | Pydantic model: `file_path`, `language`, `summary`, `key_functions`, `complexity_rating` |

### Layer 1 — Frontend & UI Design (`serving/frontend/static/`)

An enterprise-grade workspace built with open-source design system
components (Cloudscape/Ant Design/Fluent-inspired tokens).  Provides a
collapsible-sidebar layout, status chips, multi-terminal viewports, and
gesture/voice/3D-AR interaction patterns for direct manipulation.

| File | Description |
|---|---|
| `index.html` | Workspace layout: sidebar tool palette, status header, terminal panels, chat viewport |
| `design-system.css` | Enterprise design tokens (colors, spacing, typography, component variants) |
| `workspace.css` | Layout-specific styling for collapsible panels and viewports |
| `workspace.js` | WebSocket client + UI event handling (drag-and-drop, voice triggers) |

## Testing

1766 tests across the full project, including 198 intelligence module tests in
`tests/test_intelligence.py`, 98 runtime tests in `tests/test_runtime.py`.
Platform-specific tests in `tests/test_platform.py` (1476 tests) and new agent
harness tests in `tests/test_sw_platform.py` (92 tests):

- **CapabilitySchema** (4): creation, defaults, permission matching
- **CapabilityRegistry** (14): register/unregister, get/list/search,
  enable/disable, system_prompt, parse_calls, execute_call, format_results
- **CapabilityDiscovery** (3): task matching, no match, disabled exclusion
- **SessionState** (2): creation, user_id
- **RequestContext** (5): creation, user message, tool results, assistant
- **ContextBuilder** (3): from_request, build_system_prompt, empty
- **PermissionLevel** (3): ordering, numeric, value
- **PermissionPolicy** (2): default, with level
- **PermissionEvaluator** (8): allowed, insufficient, disabled, denied,
  whitelist, sandbox needs, max level
- **SandboxExecutor** (8): success, error, timeout, path check (3), file size (2)
- **AuditEvent** (2): creation, to_dict
- **AuditLog** (5): record/query, recent, overflow, clear, query by status
- **ExecutionLoop** (4): no calls, single call, max_steps, permission denied
- **Orchestrator** (12): simple, tool call, multi-round, permission denied,
  fallback, plain text, list_capabilities, to_dict, elapsed, request_id,
  audit trail, round limit
- **Public API** (1): all imports work
- **Agent harness + tools + coder** (92): see `tests/test_sw_platform.py` —
  ExecutionResult (4), ToolSpec (2), CodeExecutionProvider (9),
  FilesystemProvider (5), DatabaseProvider (3), GitProvider (6),
  WebAutomationProvider (3), RepoContext (10), StructuredOutput (4),
  DockerSandbox (5), CoderProvider (8), AgentHarness (8), create_harness_agent (3)
- **Intelligence modules** (198): see `tests/test_intelligence.py` —
  Transformers (34: softmax, attention, MHA, masked MHA, positional encoding,
  encoder/decoder layers, full Transformer, BPE tokenizer),
  Embeddings (28: tokenize, cosine similarity, normalize, TFIDF, WordEmbedding,
  SentenceEmbedding, hash_vectorize),
  Vector DB (25: VectorStore, HybridSearch, VectorIndex, VectorEntry),
  MCP (16: server tools/resources/prompts, client, handle_request),
  PEFT (17: LoRA layer, adapter, trainer, estimation functions),
  Prompt (12: PromptTemplate, FewShotBuilder, ChainOfThought,
  StructuredPrompt, PromptVariant, PromptOptimizer),
  RAG (15: Chunker, Retriever, RAGPromptBuilder, RAGPipeline),
  Multimodal (17: Image, ImageEncoder, Audio, AudioEncoder, MultimodalEncoder),
  Observability (18: TraceProvider, MetricRegistry, Guardrail, RedTeam, Span)

Run: `pytest tests/test_platform.py -v` and `pytest tests/test_sw_platform.py -v`

## Intelligence layer

26 cognitive modules in `intelligence/`, organized into Gen AI roadmap stages:

### Gen AI roadmap — intelligence modules

| Module | Tests | Roadmap Steps | Topics |
|--------|-------|---------------|--------|
| `intelligence.transformers` | 34 | 01–02 | Softmax, scaled dot-product attention, MultiHeadAttention, MaskedMultiHeadAttention, SinusoidalPositionalEncoding, LearnedPositionalEncoding, TransformerEncoder/Decoder layers, full Transformer, BPE tokenizer |
| `intelligence.embeddings` | 28 | 05–06 | Tokenization, cosine similarity, L2 normalization, TF-IDF embedder, WordEmbedding, SentenceEmbedding, feature hashing |
| `intelligence.vector_db` | 25 | 05–06 | VectorStore (CRUD + search), VectorIndex (clustering), HybridSearch (dense + sparse fusion via RRF) |
| `intelligence.mcp` | 16 | 07–09 | MCPServer (tools/resources/prompts, JSON-RPC 2.0), MCPClient (in-process + stdio), MCPTool/MCPResource/MCPPrompt dataclasses |
| `intelligence.peft` | 17 | 10–11 | LoRALayer, LoRAAdapter, LoRATrainer, LoRAConfig, merge/unmerge, parameter & memory estimation |
| `intelligence.prompt` | 12 | 03 | PromptTemplate, FewShotBuilder, ChainOfThought, StructuredPrompt, PromptVariant, PromptOptimizer (A/B testing) |
| `intelligence.rag` | 15 | 05–06 | Chunker (sentence/fixed/paragraph), Retriever, RAGPromptBuilder, RAGPipeline (ingest→retrieve→augment→generate) |
| `intelligence.multimodal` | 17 | 10–11 | Image (resize, grayscale, patches, normalization), ImageEncoder, Audio (resample, spectrogram, mel-spectrogram), AudioEncoder, MultimodalEncoder (cross-modal fusion) |
| `intelligence.observability` | 18 | 14–16 | TraceProvider (spans, traces), MetricRegistry (counters, gauges, histograms), Guardrail (toxicity/PII/blocking), RedTeam, Span |

All 9 Gen AI modules are stdlib + numpy only — no torch at import time.
All 198 intelligence tests are in `tests/test_intelligence.py`.

### Foundational cognitive modules

17 foundational modules, each stdlib-only (no numpy, no torch):

Foundation modules in `foundation/` with tests:

| Module | Tests | Topics |
|--------|-------|--------|
| `training` | 47 | TrainConfig, scheduler, checkpoint, optimizer, pretraining data, training loop |
| `curriculum` | 15 | StageConfig, CurriculumConfig, multi-stage training |
| `evaluation` | 29 | EvalConfig, EvalReport, EvalSuite, perplexity, benchmarks |
| `database` | 22 | SQLite experiment tracking |
| `alignment` | 12 | Preference training, reward modeling, DPO |
| `sft` | 7 | Supervised fine-tuning config and trainer |
| `inference` | 19 | Generator, InferenceConfig, GenerationResult |
| `model` | 15 | ModelConfig, layers, SilverwingDecoder |
| `tokenizer` | 10 | TokenizerV2 training and inference |
| `corpus` | 37 | Corpus building, integrity, dataset hashing |
| `math_corpus` | 14 | Math-specific corpus generation |
| `math_benchmark` | 5 | Math benchmark evaluation |

---

## Legacy: `runtime/` package

The original `runtime/` package is preserved for reference. It provides
the same functionality with a simpler API (no permission levels, no audit
events, no schema versioning). All 98 tests are in `tests/test_runtime.py`.
