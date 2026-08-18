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
   Corpus                           Mathematics
   Tokenizer                        Reasoning
   Model                            Engineering
   Training                         Memory
   Evaluation                       Planning
   Alignment                         Tools
   Inference
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

## Top-level layout

| Path            | Purpose                                              |
|-----------------|------------------------------------------------------|
| `foundation/`   | corpus, tokenizer, model, training, evaluation, alignment, inference |
| `intelligence/` | mathematics, reasoning, engineering, memory, planning, tools         |
| `serving/`      | api, gateway, runtime                                 |
| `benchmarks/`   | math, reasoning, engineering, language, regression    |
| `configs/`      | versioned experiment configurations                   |
| `experiments/`  | run manifests and logs (see `experiments/README.md`)  |
| `scripts/`      | operational scripts                                   |
| `tests/`        | test suite                                            |
| `docs/`         | design and process documentation                      |
| `legacy/`       | frozen prototype curriculum (phase1–phase5, math_training) |

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
