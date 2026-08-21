"""Database layer for experiment tracking and metadata storage.

Provides SQLite-backed persistence with migrations, context managers,
backup/restore, query builder, metrics aggregation, and regression detection.

- **Experiment metadata** — run configs, hyperparameters, metrics, lineage
- **Corpus records** — provenance, lineage, statistics
- **Benchmark results** — scores, comparisons, regression gates
- **Model checkpoints** — paths, hashes, training state
- **Audit log** — automatic tracking of all mutations
- **Migrations** — versioned schema evolution
"""

from .store import (
    BenchmarkRecord,
    CheckpointRecord,
    CorpusRecord,
    Database,
    ExperimentRecord,
    MigrationRecord,
    QueryBuilder,
    RegressionResult,
)

__all__ = [
    "Database",
    "ExperimentRecord",
    "CheckpointRecord",
    "BenchmarkRecord",
    "CorpusRecord",
    "MigrationRecord",
    "RegressionResult",
    "QueryBuilder",
]
