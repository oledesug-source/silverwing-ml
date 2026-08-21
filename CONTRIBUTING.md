# Contributing to Silverwing-ML

Thank you for your interest in contributing to Silverwing-ML! This document
describes the development workflow, coding standards, and how to get your
changes reviewed and merged.

## Quick start

```powershell
# Clone and set up
git clone <repo-url>
cd Silverwing-ML

# (Optional) Create a virtual environment — see .env.example
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the test suite
python -m pytest tests/ -q
```

## Development workflow

1. **Clone the repo** and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Write code** following the project conventions (see below).

3. **Run the linters** before pushing:
   ```bash
   ruff check .
   ruff format --check .
   ```

4. **Run tests** to ensure nothing breaks:
   ```bash
   python -m pytest tests/ -q
   ```

5. **Open a pull request** with a clear description of the change.

## Coding standards

- **Python 3.11+** (the project uses `from __future__ import annotations` and
  `dict[str, ...]` style type hints).
- **Ruff** is the linter and formatter. Run `ruff check .` and `ruff format .`
  before committing.
- **No external dependencies** in `intelligence/` modules — they are stdlib-only
  by design. The `foundation/` layer may use `numpy`, `torch`, and `datasets`.
- **Type hints** are encouraged but not strictly required everywhere yet (mypy
  is configured but not yet enforced in CI).
- **`__init__.py` files** should have a one-line module docstring describing
  the package.

## M01 rules (project-specific)

- Every training run must trace to a git commit hash.
- Every dataset must have a recorded content hash.
- Every config change must be versioned and committed.
- The pipeline rejects an empty corpus release by default.

## Directory structure

| Directory | Purpose |
|-----------|---------|
| `foundation/` | Core ML pipeline: corpus, tokenizer, model, training, evaluation, alignment, inference, database, curriculum, SFT, reasoning, math corpus |
| `intelligence/` | 17 cognitive modules (stdlib-only, no external deps) |
| `sw_platform/` | Controlled intelligence platform (orchestration, sandbox, capabilities) |
| `serving/` | Model serving API and runtime |
| `runtime/` | Legacy runtime (capabilities, orchestration, sandbox) |
| `benchmarks/` | Math, reasoning, engineering, regression benchmarking |
| `configs/` | Versioned experiment configurations |
| `experiments/` | Run manifests and logs |
| `scripts/` | Operational CLI scripts |
| `tests/` | Full test suite (1476 tests) |
| `dataset_gen/` | Dataset generation utilities |
| `docs/` | Design and process documentation |
| `legacy/` | Frozen prototype curriculum |

## Adding a new intelligence module

1. Create `intelligence/<module_name>/` with an `__init__.py`.
2. Add test file `tests/test_intelligence_<module_name>.py`.
3. Update the module table in `README.md`.
4. Run `python -m pytest tests/ -q` to verify.

## Code of conduct

Be respectful and constructive in all interactions. The project is committed
to providing a welcoming and safe environment for all contributors.
