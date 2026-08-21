"""SQLite-backed experiment tracking database with migrations, backup,
query builder, metrics aggregation, and regression detection.

Stores all metadata needed to reproduce experiments, track lineage,
and gate regressions.  The database is idempotent — calling
``Database.initialize()`` multiple times is safe.

Usage::

    db = Database("experiments/metadata.db")
    db.initialize()

    with db:
        exp_id = db.insert_experiment(
            ExperimentRecord(name="tokenizer-v2", config_hash="abc123")
        )
        db.insert_checkpoint(
            CheckpointRecord(experiment_id=exp_id, step=100, path="model.pt")
        )
        best = db.get_best_experiment("math-v1", higher_is_better=True)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """A single experiment run."""

    name: str
    config_hash: str
    status: str = "pending"
    experiment_id: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_id: int | None = None


@dataclass
class CheckpointRecord:
    """A saved model checkpoint."""

    experiment_id: int
    step: int
    path: str
    file_hash: str = ""
    size_bytes: int = 0
    created_at: float | None = None
    checkpoint_id: int | None = None


@dataclass
class BenchmarkRecord:
    """A benchmark evaluation result."""

    experiment_id: int
    benchmark_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    created_at: float | None = None
    benchmark_id: int | None = None


@dataclass
class CorpusRecord:
    """A corpus build record with provenance."""

    name: str
    source_count: int
    total_tokens: int
    config_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: float | None = None
    corpus_id: int | None = None


@dataclass
class MigrationRecord:
    """A schema migration record."""

    version: int
    name: str
    applied_at: float = field(default_factory=time.time)


@dataclass
class RegressionResult:
    """Result of a regression check between experiments."""

    benchmark_name: str
    baseline_experiment: str
    baseline_score: float
    candidate_experiment: str
    candidate_score: float
    delta: float
    regressed: bool
    threshold: float = 0.01


# ---------------------------------------------------------------------------
# Schema & Migrations
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "initial_schema", """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    started_at    REAL,
    finished_at   REAL,
    metrics       TEXT DEFAULT '{}',
    tags          TEXT DEFAULT '[]',
    parent_id     INTEGER,
    FOREIGN KEY (parent_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    step          INTEGER NOT NULL,
    path          TEXT NOT NULL,
    file_hash     TEXT DEFAULT '',
    size_bytes    INTEGER DEFAULT 0,
    created_at    REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    benchmark_name TEXT NOT NULL,
    score         REAL NOT NULL,
    details       TEXT DEFAULT '{}',
    created_at    REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS corpora (
    corpus_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    source_count  INTEGER NOT NULL,
    total_tokens  INTEGER NOT NULL,
    config_hash   TEXT DEFAULT '',
    provenance    TEXT DEFAULT '{}',
    created_at    REAL
);
"""),
    (2, "add_experiment_name_index", """
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
"""),
    (3, "add_audit_log", """
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    action     TEXT NOT NULL,
    entity     TEXT NOT NULL,
    entity_id  INTEGER,
    details    TEXT DEFAULT '{}',
    created_at REAL
);
"""),
    (4, "add_tags_index", """
CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON benchmarks(benchmark_name);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);
"""),
]


# ---------------------------------------------------------------------------
# Query Builder
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Fluent query builder for experiment searches.

    Usage::

        results = (
            QueryBuilder(db)
            .experiments()
            .where(status="running")
            .where(tags=["gpu"])
            .order_by("started_at", desc=True)
            .limit(10)
            .execute()
        )
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._table: str = ""
        self._conditions: list[str] = []
        self._params: list[Any] = []
        self._order: str = ""
        self._limit_val: int | None = None
        self._offset_val: int = 0

    def experiments(self) -> QueryBuilder:
        self._table = "experiments"
        return self

    def benchmarks(self) -> QueryBuilder:
        self._table = "benchmarks"
        return self

    def corpora(self) -> QueryBuilder:
        self._table = "corpora"
        return self

    def checkpoints(self) -> QueryBuilder:
        self._table = "checkpoints"
        return self

    def where(self, **kwargs: Any) -> QueryBuilder:
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, list):
                placeholders = ",".join("?" for _ in value)
                self._conditions.append(f"{key} IN ({placeholders})")
                self._params.extend(value)
            elif isinstance(value, bool):
                self._conditions.append(f"{key} = ?")
                self._params.append(1 if value else 0)
            elif isinstance(value, str) and value.startswith("%"):
                self._conditions.append(f"{key} LIKE ?")
                self._params.append(value)
            else:
                self._conditions.append(f"{key} = ?")
                self._params.append(value)
        return self

    def where_gt(self, column: str, value: Any) -> QueryBuilder:
        self._conditions.append(f"{column} > ?")
        self._params.append(value)
        return self

    def where_lt(self, column: str, value: Any) -> QueryBuilder:
        self._conditions.append(f"{column} < ?")
        self._params.append(value)
        return self

    def order_by(self, column: str, desc: bool = False) -> QueryBuilder:
        direction = "DESC" if desc else "ASC"
        self._order = f" ORDER BY {column} {direction}"
        return self

    def limit(self, n: int) -> QueryBuilder:
        self._limit_val = n
        return self

    def offset(self, n: int) -> QueryBuilder:
        self._offset_val = n
        return self

    def execute(self) -> list[dict[str, Any]]:
        """Execute the built query and return results."""
        if not self._table:
            raise ValueError("No table selected. Call .experiments(), etc.")

        where = ""
        if self._conditions:
            where = " WHERE " + " AND ".join(self._conditions)

        sql = f"SELECT * FROM {self._table}{where}{self._order}"
        if self._limit_val is not None:
            sql += f" LIMIT {self._limit_val}"
        if self._offset_val > 0:
            sql += f" OFFSET {self._offset_val}"

        rows = self._db.conn.execute(sql, self._params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """SQLite experiment tracking database with migrations, context manager,
    backup/restore, query builder, metrics aggregation, and regression detection.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create tables and run pending migrations."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._run_migrations()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        if self._conn is None:
            self.initialize()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.initialize()
        assert self._conn is not None
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for a database transaction with auto-commit/rollback."""
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _run_migrations(self) -> None:
        """Apply all pending migrations."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version  INTEGER PRIMARY KEY,
                name     TEXT NOT NULL,
                applied_at REAL
            )
        """)
        self.conn.commit()

        applied = {
            row[0]
            for row in self.conn.execute("SELECT version FROM schema_migrations")
        }

        for version, name, sql in _MIGRATIONS:
            if version not in applied:
                self.conn.executescript(sql)
                self.conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, time.time()),
                )
                self.conn.commit()

    def get_migration_version(self) -> int:
        """Return the current schema version."""
        try:
            row = self.conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
            return row[0] if row and row[0] else 0
        except sqlite3.OperationalError:
            return 0

    # ------------------------------------------------------------------
    # Backup / Restore
    # ------------------------------------------------------------------

    def backup(self, dest: str | Path) -> Path:
        """Create a file-level backup of the database."""
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if self._conn:
            self._conn.commit()
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        shutil.copy2(self._path, dest_path)
        for suffix in ("-wal", "-shm"):
            src = Path(str(self._path) + suffix)
            if src.exists():
                shutil.copy2(src, Path(str(dest_path) + suffix))
        return dest_path

    def restore(self, source: str | Path) -> None:
        """Restore database from a backup file."""
        self.close()
        shutil.copy2(source, self._path)
        for suffix in ("-wal", "-shm"):
            src = Path(str(source) + suffix)
            dst = Path(str(self._path) + suffix)
            if src.exists():
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()
        self.initialize()

    def vacuum(self) -> None:
        """Reclaim unused space."""
        self.conn.execute("VACUUM")

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def insert_experiment(self, rec: ExperimentRecord) -> int:
        """Insert an experiment record and return its ID."""
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO experiments
               (name, config_hash, status, started_at, metrics, tags, parent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.name,
                rec.config_hash,
                rec.status,
                rec.started_at or now,
                json.dumps(rec.metrics),
                json.dumps(rec.tags),
                rec.parent_id,
            ),
        )
        self.conn.commit()
        self._audit("insert", "experiment", cur.lastrowid)
        return cur.lastrowid  # type: ignore[return-value]

    def update_experiment(
        self,
        experiment_id: int,
        *,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        finished_at: float | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Update fields of an existing experiment."""
        updates: list[str] = []
        values: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if metrics is not None:
            updates.append("metrics = ?")
            values.append(json.dumps(metrics))
        if finished_at is not None:
            updates.append("finished_at = ?")
            values.append(finished_at)
        if tags is not None:
            updates.append("tags = ?")
            values.append(json.dumps(tags))
        if not updates:
            return
        values.append(experiment_id)
        self.conn.execute(
            f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?",
            values,
        )
        self.conn.commit()
        self._audit("update", "experiment", experiment_id)

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        """Retrieve an experiment by ID."""
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_experiments(self, status: str | None = None) -> list[dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE status = ? ORDER BY experiment_id DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiments ORDER BY experiment_id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_experiments(self, status: str | None = None) -> int:
        """Count experiments, optionally filtered by status."""
        if status:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        return row[0] if row else 0

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment and its associated records."""
        with self.transaction():
            self.conn.execute(
                "DELETE FROM benchmarks WHERE experiment_id = ?", (experiment_id,)
            )
            self.conn.execute(
                "DELETE FROM checkpoints WHERE experiment_id = ?", (experiment_id,)
            )
            cur = self.conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
            )
        self._audit("delete", "experiment", experiment_id)
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def insert_checkpoint(self, rec: CheckpointRecord) -> int:
        """Insert a checkpoint record and return its ID."""
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO checkpoints
               (experiment_id, step, path, file_hash, size_bytes, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rec.experiment_id,
                rec.step,
                rec.path,
                rec.file_hash,
                rec.size_bytes,
                rec.created_at or now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_checkpoints(self, experiment_id: int) -> list[dict[str, Any]]:
        """List checkpoints for an experiment, ordered by step."""
        rows = self.conn.execute(
            "SELECT * FROM checkpoints WHERE experiment_id = ? ORDER BY step",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_checkpoint(self, experiment_id: int) -> dict[str, Any] | None:
        """Get the most recent checkpoint for an experiment."""
        row = self.conn.execute(
            "SELECT * FROM checkpoints WHERE experiment_id = ? ORDER BY step DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def insert_benchmark(self, rec: BenchmarkRecord) -> int:
        """Insert a benchmark result and return its ID."""
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO benchmarks
               (experiment_id, benchmark_name, score, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                rec.experiment_id,
                rec.benchmark_name,
                rec.score,
                json.dumps(rec.details),
                rec.created_at or now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_benchmark_history(
        self, benchmark_name: str
    ) -> list[dict[str, Any]]:
        """Get all results for a named benchmark, newest first."""
        rows = self.conn.execute(
            """SELECT b.*, e.name as experiment_name
               FROM benchmarks b
               JOIN experiments e ON b.experiment_id = e.experiment_id
               WHERE b.benchmark_name = ?
               ORDER BY b.created_at DESC""",
            (benchmark_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_experiment(
        self, benchmark_name: str, higher_is_better: bool = True
    ) -> dict[str, Any] | None:
        """Get the experiment with the best score for a benchmark."""
        direction = "DESC" if higher_is_better else "ASC"
        row = self.conn.execute(
            f"""SELECT e.*, b.score, b.benchmark_name
                FROM experiments e
                JOIN benchmarks b ON e.experiment_id = b.experiment_id
                WHERE b.benchmark_name = ?
                ORDER BY b.score {direction}
                LIMIT 1""",
            (benchmark_name,),
        ).fetchone()
        return dict(row) if row else None

    def detect_regression(
        self,
        benchmark_name: str,
        baseline_experiment: str,
        candidate_experiment: str,
        threshold: float = 0.01,
    ) -> RegressionResult | None:
        """Detect if candidate regressed compared to baseline."""
        rows = self.conn.execute(
            """SELECT e.name, b.score
               FROM benchmarks b
               JOIN experiments e ON b.experiment_id = e.experiment_id
               WHERE b.benchmark_name = ? AND e.name IN (?, ?)""",
            (benchmark_name, baseline_experiment, candidate_experiment),
        ).fetchall()

        scores = {row[0]: row[1] for row in rows}
        if baseline_experiment not in scores or candidate_experiment not in scores:
            return None

        baseline_score = scores[baseline_experiment]
        candidate_score = scores[candidate_experiment]
        delta = candidate_score - baseline_score

        return RegressionResult(
            benchmark_name=benchmark_name,
            baseline_experiment=baseline_experiment,
            baseline_score=baseline_score,
            candidate_experiment=candidate_experiment,
            candidate_score=candidate_score,
            delta=delta,
            regressed=delta < -threshold,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Corpora
    # ------------------------------------------------------------------

    def insert_corpus(self, rec: CorpusRecord) -> int:
        """Insert a corpus record and return its ID."""
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO corpora
               (name, source_count, total_tokens, config_hash, provenance, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rec.name,
                rec.source_count,
                rec.total_tokens,
                rec.config_hash,
                json.dumps(rec.provenance),
                rec.created_at or now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_corpora(self) -> list[dict[str, Any]]:
        """List all corpora, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM corpora ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Audit Log
    # ------------------------------------------------------------------

    def _audit(self, action: str, entity: str, entity_id: int | None = None) -> None:
        """Write an audit log entry."""
        try:
            self.conn.execute(
                """INSERT INTO audit_log (action, entity, entity_id, created_at)
                   VALUES (?, ?, ?, ?)""",
                (action, entity, entity_id, time.time()),
            )
        except sqlite3.OperationalError:
            pass

    def get_audit_log(
        self, entity: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve audit log entries."""
        if entity:
            rows = self.conn.execute(
                "SELECT * FROM audit_log WHERE entity = ? ORDER BY audit_id DESC LIMIT ?",
                (entity, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Metrics Aggregation
    # ------------------------------------------------------------------

    def aggregate_benchmarks(
        self, benchmark_name: str
    ) -> dict[str, Any]:
        """Compute aggregate statistics for a benchmark."""
        row = self.conn.execute(
            """SELECT
                COUNT(*) as count,
                AVG(score) as mean,
                MIN(score) as min_score,
                MAX(score) as max_score,
                MAX(score) - MIN(score) as range_score
               FROM benchmarks
               WHERE benchmark_name = ?""",
            (benchmark_name,),
        ).fetchone()
        if row is None or row[0] == 0:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "range": 0}
        return {
            "count": row[0],
            "mean": round(row[1], 4) if row[1] else 0,
            "min": row[2],
            "max": row[3],
            "range": round(row[4], 4) if row[4] else 0,
        }

    def get_experiment_lineage(self, experiment_id: int) -> list[dict[str, Any]]:
        """Walk the parent chain of an experiment."""
        chain: list[dict[str, Any]] = []
        current_id: int | None = experiment_id
        visited: set[int] = set()

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            exp = self.get_experiment(current_id)
            if exp:
                chain.append(exp)
                current_id = exp.get("parent_id")
            else:
                break

        return list(reversed(chain))

    def get_disk_usage(self) -> dict[str, Any]:
        """Return approximate database size and record counts."""
        size = self._path.stat().st_size if self._path.exists() else 0
        counts = {}
        for table in ("experiments", "checkpoints", "benchmarks", "corpora", "audit_log"):
            try:
                row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0] if row else 0
            except sqlite3.OperationalError:
                counts[table] = 0
        return {"size_bytes": size, "size_mb": round(size / (1024 * 1024), 2), **counts}

    # ------------------------------------------------------------------
    # Query Builder Entry Point
    # ------------------------------------------------------------------

    def query(self) -> QueryBuilder:
        """Start building a query."""
        return QueryBuilder(self)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def hash_config(config: dict[str, Any]) -> str:
        """Produce a deterministic SHA-256 hash of a config dict."""
        raw = json.dumps(config, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]
