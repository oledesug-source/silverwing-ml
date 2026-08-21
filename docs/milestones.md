# Silverwing-ML Milestone Tracker

## Project Stages

| Stage | Config | Description | Status |
|-------|--------|-------------|--------|
| M0 | `configs/foundation.yaml` | Project setup, foundation spec, training infrastructure | ✅ Done |
| M01 | `configs/tokenizer.yaml` | Tokenizer training | ✅ Done |
| M08 | `configs/math_corpus.yaml` | Math corpus generation (corpus-v1) | ✅ Done |
| M10 | `configs/training.yaml` | Initial pretraining on corpus-v1 | ✅ Done |
| M11 | `configs/sft.yaml` | Supervised fine-tuning (SFT) | ✅ Done |
| M13 | `configs/sft_combined.yaml` | Combined SFT (basic + reasoning CoT) | ✅ Done |
| M14 | `configs/corpus.yaml` | Corpus expansion (OpenWebText quickstart) | ✅ Done |

## Promotion Criteria (M01 Rule)

Every trained artifact must satisfy all of the following to be promoted:

- [x] `require_git_commit` — run traces to a specific git commit
- [x] `require_dataset_hash` — dataset content hash is recorded
- [x] `require_model_checkpoint` — model checkpoint is saved
- [x] `require_held_out_evaluation` — evaluation on held-out data
- [x] `require_zero_critical_regressions` — no critical metric regressions

## Code Quality Status

| Check | Status | Notes |
|-------|--------|-------|
| Ruff lint (`ruff check .`) | ✅ 0 errors | All rules pass (E, F, W, I, UP, B, C4) |
| Ruff format (`ruff format --check .`) | ⚠️ 188 files | Pre-existing formatting gaps, not blocking |
| Mypy (`mypy .`) | ⚠️ 434 errors | Config set up with third-party overrides; 1 import bug fixed (serving/runtime/runtime.py). 285 source-level type issues remain pre-existing. Type checking not yet enforced in CI. |
| Pytest | ✅ 1766 tests pass | 1476 platform/runtime + 92 agent harness + 198 intelligence module tests |

## Lint Rule Decisions

| Rule | Decision | Rationale |
|------|----------|-----------|
| B905 (`zip` without `strict=`) | **Disabled project-wide** | `zip()` is used pervasively throughout the codebase. Adding `strict=True` is a behavior-changing migration that risks introducing runtime `ValueError` exceptions if iterable lengths diverge unexpectedly. Deferred until an intentional audit can be performed. This decision is documented here and in `CHANGELOG.md`. |
