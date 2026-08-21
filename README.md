# Silverwing-ML

Silverwing is a from-scratch, open, self-verified language-model project.

## Architecture

```
                    SILVERWING
                        │
        ┌───────────────┴───────────────┐
        │                               │
   FOUNDATION                       INTELLIGENCE
        │                               │
   Corpus (35)                     Mathematics (10)
   Tokenizer (10)                  Reasoning (20)
   Model (15)                      Engineering (8)
   Training (47)                   Memory (20)
   Evaluation (29)                 Planning (12)
   Alignment (12)                  Tools (18)
   Inference (19)                  Operating Systems (47)
   Database (22)                   Networking (50)
   Curriculum (15)                 Security (72)
   SFT (7)                         Data Science (103)
   Reasoning Training (12)         Statistics (62)
   Math Corpus (14)                Linear Algebra (64)
                                    ML Basics (78)
                                    OOP (118)
                                    WebDev (144)
                                    Visualization (105)
                                    Training (113)
                                    Transformers (34)
                                    Embeddings (28)
                                    Vector DB (25)
                                    MCP (16)
                                    PEFT (17)
                                    Prompt (12)
                                    RAG (15)
                                    Multimodal (17)
                                    Observability (18)
        │
        └──────────────┬────────────────┘
                       │
                    SERVING
                       │
                 API / Gateway
                       │
                    Runtime
```

Underneath everything: **Git** + **Dataset Provenance** + **Experiment Tracking**
+ **Benchmarking** + **Regression Gates**.

## Four-layer platform architecture

The application stack is organized into four layers:

| Layer | Purpose | Key modules |
|---|---|---|
| **Layer 1** | Frontend & UI Design | `serving/frontend/static/` (HTML/CSS/JS enterprise workspace, design system, WebSocket client) |
| **Layer 2** | Agent Harness | `sw_platform/harness/` (pydantic_ai integration), `sw_platform/orchestration/` |
| **Layer 3** | Python Automation & Tooling | `sw_platform/tools/` (code execution, web automation, DB, filesystem, git) |
| **Layer 4** | Software & Coding Capabilities | `sw_platform/coder/` (code interpreter, repo context, Docker sandbox, structured output) |

**Layer 2 Bridge** (`serving/api/fastapi_bridge.py`): FastAPI WebSocket
and REST API that streams the agent harness to the frontend in real time.

```bash
# Run the bridge server
uvicorn serving.api.fastapi_bridge:app --reload --host 0.0.0.0 --port 8080
```

## Top-level layout

| Path            | Purpose                                              |
|-----------------|------------------------------------------------------|
| `foundation/`   | corpus, tokenizer, model, training, evaluation, alignment, inference, database, curriculum, sft, reasoning, math_corpus |
| `intelligence/` | 26 cognitive modules (9 Gen AI roadmap stages + 17 foundational) |
| `serving/`      | api, gateway, runtime                                 |
| `sw_platform/`  | controlled intelligence platform (M15 orchestration)  |
| `runtime/`      | legacy intelligence runtime (Capability, Orchestrator, Agent, Sandbox) |
| `benchmarks/`   | math, reasoning, engineering, language, regression    |
| `configs/`      | versioned experiment configurations                   |
| `experiments/`  | run manifests and logs (see `experiments/README.md`)  |
| `scripts/`      | operational scripts                                   |
| `tests/`        | 1766 tests across all modules                         |
| `docs/`         | design and process documentation                      |
| `datasets/`     | dataset generation utilities and processed data       |
| `legacy/`       | frozen prototype curriculum (phase1–phase5, math_training) |

## Intelligence modules

26 cognitive modules organized into two groups: 17 foundational modules
(stdlib-only) plus 9 Gen AI roadmap modules (stdlib + numpy). All are tested
in `tests/test_intelligence.py` (198 tests).

### Gen AI roadmap modules (Steps 01–16)

| Module | Tests | Roadmap Steps | Description |
|--------|-------|---------------|-------------|
| `intelligence.transformers` | 34 | 01–02 | Softmax, attention, MultiHeadAttention, positional encoding, Transformer, BPE tokenizer |
| `intelligence.embeddings` | 28 | 05–06 | TF-IDF, WordEmbedding, SentenceEmbedding, cosine similarity, feature hashing |
| `intelligence.vector_db` | 25 | 05–06 | VectorStore, VectorIndex, HybridSearch (dense + sparse RRF fusion) |
| `intelligence.prompt` | 12 | 03 | PromptTemplate, FewShotBuilder, ChainOfThought, StructuredPrompt, PromptOptimizer |
| `intelligence.mcp` | 16 | 07–09 | MCPServer (JSON-RPC 2.0), MCPClient, tools/resources/prompts |
| `intelligence.rag` | 15 | 05–06 | Chunker, Retriever, RAGPromptBuilder, RAGPipeline |
| `intelligence.peft` | 17 | 10–11 | LoRALayer, LoRAAdapter, LoRATrainer, LoRAConfig, parameter estimation |
| `intelligence.multimodal` | 17 | 10–11 | Image/Audio encoders, MultimodalEncoder (cross-modal fusion) |
| `intelligence.observability` | 18 | 14–16 | TraceProvider, MetricRegistry, Guardrail, RedTeam, Span |

### Foundational cognitive modules

17 foundational modules, each stdlib-only (no numpy, no torch):

| Module | Tests | Description |
|--------|-------|-------------|
| `operating_systems` | 47 | Processes, memory, synchronization, file I/O, CPU monitoring |
| `networking` | 50 | HTTP client/server, load balancing, rate limiting, retry, middleware, connection pools |
| `security` | 72 | Authentication, RBAC, JWT, encryption, validation, audit logging |
| `data_science` | 103 | Algorithms, data structures, graph theory, dynamic programming |
| `statistics` | 62 | Distributions, inference, regression, time series, probability |
| `linear_algebra` | 64 | Matrices, decompositions, solvers, sparse matrices |
| `ml_basics` | 78 | Linear models, trees, clustering, neighbors, preprocessing, metrics |
| `oop` | 118 | Encapsulation, inheritance, polymorphism, design patterns, metaclasses |
| `webdev` | 144 | Routing, middleware, templates, ORM, WebSocket, auth, static files |
| `visualization` | 105 | Text charts, SVG, HTML/Chart.js, color scales, reports |
| `training` | 113 | Neural networks, backpropagation, optimizers, loss functions, data loading |
| `mathematics` | 10 | Mathematical problem solving with chain-of-thought |
| `reasoning` | 20 | Logical reasoning and inference |
| `engineering` | 8 | Code understanding and generation |
| `memory` | 20 | Context window management and retrieval |
| `planning` | 12 | Task decomposition and goal planning |
| `tools` | 18 | Tool-use protocol for external system interaction |

## Milestones

| #  | Milestone                                   |
|----|---------------------------------------------|
| 01 | Repository + Reproducibility                |
| 02 | Corpus + Provenance Pipeline                |
| 03 | Deduplication + Contamination Engine        |
| 04 | Real Benchmark Engine                       |
| 05 | Tokenizer V2                               |
| 06 | Scalable Decoder V2                        |
| 07 | Training Engine V2                         |
| 08 | Mathematical Training Corpus               |
| 09 | Mathematics Benchmark                      |
| 10 | Real Pretraining                           |
| 11 | SFT                                        |
| 12 | Preference Alignment                       |
| 13 | Reasoning Training                         |
| 14 | Native Inference                           |
| 15 | Cognitive Integration                      |
| 16 | Continual Learning                         |

## Current operational workflow

The completed foundation milestones are exposed through small, versioned
scripts. Build a corpus before training its tokenizer and model:

```powershell
python scripts/build_corpus.py --source local-data=path/to/source --output-dir experiments/corpus-v1
python scripts/train_tokenizer.py --corpus-dir experiments/corpus-v1 --output-dir experiments/tokenizer-v1
python scripts/generate_math_benchmark.py --corpus-dir experiments/corpus-v1
python scripts/train.py --config configs/training.yaml
python scripts/verify_corpus.py --output-dir experiments/corpus-v1
python scripts/run_benchmark.py --benchmark math-benchmark-v1 --model dummy
```

For a production corpus, declare full source metadata in `configs/corpus.yaml`
instead of passing `--source`; the configuration is then included in the
pipeline report digest. The builder rejects an empty release by default; use
`--allow-empty` only for deliberate pipeline smoke tests.

## Legacy layer

`legacy/` contains the historical prototype: phase1–phase5 lessons and the
math_training track. It is frozen and preserved for its reusable components
(66R tokenizer, 71R decoder architecture, 73R pretraining, 77R/78R SFT,
79R/80R reasoning training, Phase 4 API/agents, verification and governance
lessons). The procedural lessons are treated as validated components and
research history — **not** as learned model intelligence.
