# Changelog

All notable changes to Silverwing-ML are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **fp16/bf16 mixed-precision training (`amp`, `amp_dtype`)** in `TrainConfig`/`SftConfig` and both trainers — `torch.autocast` + `GradScaler`, opt-in, CUDA-only (validated in `__post_init__`); CLI flags `--amp` on `scripts/train.py` and `scripts/train_sft.py`; recorded in run reports
- `colab_production.ipynb` — production manufacturing pipeline for free-tier GPUs: real-data corpus (FineWeb-Edu 400K + Wikipedia 100K), tokenizer v2 retraining, Drive-backed checkpoints with auto-resume across Colab disconnects, SFT, generation test, benchmark gate
- `--skip-existing` flag on `scripts/ingest_external_v2.py` download phase — resumes interrupted corpus builds without re-downloading
- `fineweb` preset in `configs/external_sources.yaml` (`HuggingFaceFW/fineweb-edu`, parquet streaming)
- `.env.example` — template for environment variables used by the corpus pipeline and serving API
- `requirements-dev.txt` — development dependencies (pytest-cov, ruff, mypy, pdoc3, pydantic-ai, fastapi, uvicorn, httpx)
- `CONTRIBUTING.md` — developer workflow, coding standards, and directory structure guide
- `docs/milestones.md` — milestone tracker (M01–M16) with promotion criteria
- `datasets/__init__.py` docstring
- `.gitkeep` files in `datasets/raw/` and `static/dist/` to preserve empty directories
- **Layer 1 — Frontend**: `serving/frontend/static/` enterprise workspace UI (index.html, design-system.css, workspace.css, workspace.js) with collapsible sidebars, status chips, multi-terminal viewports, and WebSocket client
- **Layer 2 — Agent Harness**: `sw_platform/harness/` (`core.py`, `agent.py`) — `PydanticAgentHarness`, `HarnessConfig`, `AgentResponse`, `ToolCallRecord`, `create_harness_agent`, `ToolProvider` protocol
- **Layer 3 — Python Automation Tooling**: `sw_platform/tools/` (`code_execution.py`, `database.py`, `filesystem.py`, `git.py`, `web_automation.py`) — 5 tool providers covering code execution, SQLite queries, filesystem ops, git commands, and web automation
- **Layer 4 — Software & Coding Capabilities**: `sw_platform/coder/` (`core.py`, `models.py`) — `CoderProvider`, `RepoContext`, `DockerSandbox`, `StructuredOutput`, `CodePatch`, `CodeExplanation`
- **FastAPI WebSocket Bridge**: `serving/api/fastapi_bridge.py` — REST + WebSocket endpoints streaming agent harness to frontend
- **92 new tests** in `tests/test_sw_platform.py` covering harness agent, all tool providers, and coder module
- **9 Gen AI roadmap intelligence modules** with 198 tests in `tests/test_intelligence.py`:
  - `intelligence.transformers/` — softmax, attention, MultiHeadAttention, MaskedMultiHeadAttention, positional encoding, Transformer encoder/decoder, BPE tokenizer (34 tests)
  - `intelligence.embeddings/` — TF-IDF, WordEmbedding, SentenceEmbedding, cosine similarity, hash vectorization (28 tests)
  - `intelligence.vector_db/` — VectorStore, VectorIndex, HybridSearch with RRF fusion (25 tests)
  - `intelligence.prompt/` — PromptTemplate, FewShotBuilder, ChainOfThought, StructuredPrompt, PromptOptimizer (12 tests)
  - `intelligence.mcp/` — MCPServer (JSON-RPC 2.0), MCPClient, tools/resources/prompts (16 tests)
  - `intelligence.rag/` — Chunker, Retriever, RAGPromptBuilder, RAGPipeline (15 tests)
  - `intelligence.peft/` — LoRALayer, LoRAAdapter, LoRATrainer, LoRAConfig, parameter estimation (17 tests)
  - `intelligence.multimodal/` — Image, Audio encoders, MultimodalEncoder (17 tests)
  - `intelligence.observability/` — TraceProvider, MetricRegistry, Guardrail, RedTeam, Span (18 tests)

### Changed
- `wikipedia` and `all` presets migrated to reliable streaming sources: `wikimedia/wikipedia` (parquet) replaces the deprecated script-based loader; `all` preset now = FineWeb-Edu + Wikipedia production mix
- Updated `README.md`, `docs/architecture.md`, `docs/milestones.md`, and `CHANGELOG.md` to reflect 1766 total test count, 9 new Gen AI roadmap intelligence modules (198 tests), and updated package layout
- `.gitignore` updated with `!.env.example` negation and `.env` exclusion; raw ingested corpora (`experiments/ingested/`) and `mlflow.db` now ignored; root-level `platform/`/`static/` ignore rules scoped so `serving/frontend/static/` stays tracked
- `pyproject.toml` mypy config: added `follow_imports`, `ignore_missing_imports` overrides for `torch.*`, `datasets.*`, `yaml.*`, `psutil.*`, `transformers.*`, `numpy.*`
- `pyproject.toml` mypy `python_version` bumped from `3.11` to `3.12` (numpy 2.x bundled stubs use PEP 695 `type` statement syntax requiring Python 3.12+)
- `.github/workflows/ci.yml` updated: both `test` and `lint` jobs now install from `requirements-dev.txt`

### Fixed
- Broken relative import in `serving/runtime/runtime.py` — `from .api.server` → `from ..api.server` (SilverwingHandler lives in `serving/api/server.py`)
- Added `serving/gateway/__init__.py` and `.gitkeep` to preserve the gateway module
- 8 remaining ruff lint errors resolved:
  - **B904** — `raise ImportError(...)` in `foundation/corpus/ingestion.py` now uses `from None`
  - **B024** (×2) — `Flyer(ABC)` and `Swimmer(ABC)` in `intelligence/oop/inheritance.py` no longer inherit `ABC` (they are concrete mixins)
  - **B007** (×3) — Unused loop variables renamed with `_` prefix in `charts.py` (RadarChart) and `svg.py` (PieChartSVG)
  - **UP031** — `%` formatting in `tests/test_benchmarks.py` converted to f-string
  - **F811** — Removed duplicate `mse` import in `tests/test_ml_basics.py` (was shadowing `metrics.mse`)

### Lint Rule Decisions
- **B905** (`zip` without `strict=`): Permanently disabled project-wide. The rule is too pervasive across the codebase to address without a behavior-changing migration. Adding `strict=True` to every `zip()` call risks runtime `ValueError` exceptions if iterable lengths diverge. Deferred until a deliberate, tested audit can be performed. Decision documented in `docs/milestones.md` and `CHANGELOG.md`.

## [1.1.0] — 2025-08-19

### Added
- `sw_platform/` — controlled intelligence platform (orchestration, sandbox, capability registry)
- `foundation/database/` — SQLite-based dataset store with schema versioning
- `intelligence/training/` — neural network module (Linear, MLP, Sequential, optimizers, loss functions, data loading)
- `intelligence/data_science/` — algorithms, graph theory, dynamic programming
- `intelligence/ml_basics/` — linear models, trees, clustering, neighbors, preprocessing, metrics
- `intelligence/linear_algebra/` — matrices, decompositions, solvers, sparse matrices
- `intelligence/statistics/` — distributions, inference, regression, time series
- `intelligence/networking/` — HTTP client/server, load balancing, rate limiting, middleware
- `intelligence/operating_systems/` — processes, memory, synchronization, file I/O, CPU monitoring
- `intelligence/security/` — authentication, RBAC, JWT, encryption, audit logging
- `intelligence/webdev/` — routing, middleware, templates, ORM, WebSocket, auth
- `intelligence/visualization/` — text charts, SVG, HTML/Chart.js, reports
- `intelligence/oop/` — encapsulation, inheritance, polymorphism, design patterns
- `intelligence/mathematics/` — mathematical problem solving with chain-of-thought
- `intelligence/reasoning/` — logical reasoning and inference
- `intelligence/engineering/` — code understanding and generation
- `intelligence/memory/` — context window management and retrieval
- `intelligence/planning/` — task decomposition and goal planning
- `intelligence/tools/` — tool-use protocol
- `intelligence/curriculum/` — progressive cognitive skill development
- `intelligence/memory/context.py` — per-request state containers
- `runtime/` — legacy intelligence runtime (Capability, Orchestrator, Agent, Sandbox)
- `configs/model_small.yaml` — small model config (~10M params, CPU-friendly)
- `configs/training_small.yaml` — quickstart training config
- `configs/training_quickstart.yaml` — minimal training config
- `configs/training_scaled.yaml` — scaled training config
- `configs/training_cont.yaml` — continued pretraining config
- `configs/training_cont2.yaml` — second continued pretraining config
- `configs/sft_small.yaml` — small SFT config
- `configs/sft_comprehensive.yaml` — comprehensive SFT config
- `configs/sft_combined.yaml` — combined SFT config
- `configs/sft_reasoning.yaml` — reasoning SFT config
- `configs/sft_dataset.yaml` — SFT dataset config
- `configs/reasoning_dataset.yaml` — reasoning dataset config
- `configs/reasoning.yaml` — reasoning training config
- `configs/math_benchmark.yaml` — math benchmark config
- `configs/math_corpus.yaml` — math corpus config
- `configs/external_sources.yaml` — external data source presets
- `configs/curriculum.yaml` — curriculum training config
- `configs/alignment.yaml` — alignment training config
- `benchmarks/` — math, reasoning, engineering, language, regression benchmarking
- `benchmarks/guard.py` — regression gates for metric validation
- `benchmarks/registry.py` — benchmark registry
- `benchmarks/dataset.py` — benchmark dataset utilities
- `benchmarks/metrics.py` — benchmark metrics
- `benchmarks/report.py` — evaluation report formatting
- `benchmarks/runner.py` — benchmark runner
- `tests/__init__.py` — test package initialization
- `.pre-commit-config.yaml` — pre-commit hooks (ruff format + check)
- `.github/workflows/ci.yml` — CI pipeline (ruff lint + pytest)
- `LICENSE` — Apache 2.0
- `.gitattributes` — file attribute configuration

### Changed
- Ruff configuration expanded: added C4 (comprehensions) rule set
- `pyproject.toml` updated with project metadata, optional dependencies, pytest/ruff/mypy configs
- All 1179 initial ruff lint errors reduced to 0 through automated fixes and manual corrections
