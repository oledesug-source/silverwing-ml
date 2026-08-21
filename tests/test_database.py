"""Tests for enhanced foundation.database — migrations, query builder, backup, aggregation, regression."""

import tempfile
from pathlib import Path

from foundation.database import (
    BenchmarkRecord,
    CheckpointRecord,
    Database,
    ExperimentRecord,
)


def _tmp_db() -> Database:
    path = Path(tempfile.mktemp(suffix=".db"))
    db = Database(path)
    db.initialize()
    return db


class TestContextManager:
    def test_enter_exit(self) -> None:
        path = Path(tempfile.mktemp(suffix=".db"))
        with Database(path) as db:
            db.insert_experiment(ExperimentRecord(name="ctx-test", config_hash="a"))
        assert path.exists()

    def test_transaction_commit(self) -> None:
        db = _tmp_db()
        with db.transaction():
            db.insert_experiment(ExperimentRecord(name="tx-test", config_hash="b"))
        exps = db.list_experiments()
        assert len(exps) == 1
        db.close()

    def test_transaction_rollback(self) -> None:
        db = _tmp_db()
        try:
            with db.transaction():
                db.conn.execute(
                    "INSERT INTO experiments (name, config_hash, status) VALUES (?, ?, ?)",
                    ("rollback", "c", "pending"),
                )
                raise ValueError("force rollback")
        except ValueError:
            pass
        assert len(db.list_experiments()) == 0
        db.close()


class TestMigrations:
    def test_initial_migration(self) -> None:
        db = _tmp_db()
        version = db.get_migration_version()
        assert version >= 1
        db.close()

    def test_idempotent_initialize(self) -> None:
        db = _tmp_db()
        db.initialize()
        db.initialize()
        version = db.get_migration_version()
        assert version >= 1
        db.close()

    def test_audit_log_table_exists(self) -> None:
        db = _tmp_db()
        version = db.get_migration_version()
        assert version >= 3
        db.close()


class TestQueryBuilder:
    def test_basic_query(self) -> None:
        db = _tmp_db()
        db.insert_experiment(ExperimentRecord(name="q1", config_hash="1", status="completed"))
        db.insert_experiment(ExperimentRecord(name="q2", config_hash="2", status="running"))

        results = db.query().experiments().where(status="completed").execute()
        assert len(results) == 1
        assert results[0]["name"] == "q1"
        db.close()

    def test_order_by_limit(self) -> None:
        db = _tmp_db()
        for i in range(5):
            db.insert_experiment(ExperimentRecord(name=f"e{i}", config_hash=str(i)))
        results = (
            db.query()
            .experiments()
            .order_by("experiment_id", desc=True)
            .limit(3)
            .execute()
        )
        assert len(results) == 3
        db.close()

    def test_where_in(self) -> None:
        db = _tmp_db()
        db.insert_experiment(ExperimentRecord(name="a", config_hash="1"))
        db.insert_experiment(ExperimentRecord(name="b", config_hash="2"))
        db.insert_experiment(ExperimentRecord(name="c", config_hash="3"))

        results = db.query().experiments().where(name=["a", "c"]).execute()
        assert len(results) == 2
        db.close()


class TestBackupRestore:
    def test_backup_and_restore(self) -> None:
        db = _tmp_db()
        db.insert_experiment(ExperimentRecord(name="backup-test", config_hash="x"))

        backup_path = Path(tempfile.mktemp(suffix=".bak"))
        db.backup(backup_path)
        assert backup_path.exists()

        db2_path = Path(tempfile.mktemp(suffix=".db"))
        db2 = Database(db2_path)
        db2.restore(backup_path)
        exps = db2.list_experiments()
        assert len(exps) == 1
        assert exps[0]["name"] == "backup-test"
        db.close()
        db2.close()

    def test_vacuum(self) -> None:
        db = _tmp_db()
        db.vacuum()
        db.close()


class TestAggregation:
    def test_benchmark_aggregation(self) -> None:
        db = _tmp_db()
        exp_id = db.insert_experiment(ExperimentRecord(name="agg", config_hash="a"))
        for score in [0.7, 0.8, 0.9, 0.85]:
            db.insert_benchmark(
                BenchmarkRecord(experiment_id=exp_id, benchmark_name="math-v1", score=score)
            )

        agg = db.aggregate_benchmarks("math-v1")
        assert agg["count"] == 4
        assert agg["mean"] == 0.8125
        assert agg["min"] == 0.7
        assert agg["max"] == 0.9
        db.close()

    def test_empty_aggregation(self) -> None:
        db = _tmp_db()
        agg = db.aggregate_benchmarks("nonexistent")
        assert agg["count"] == 0
        db.close()


class TestRegressionDetection:
    def test_detect_regression(self) -> None:
        db = _tmp_db()
        base_id = db.insert_experiment(ExperimentRecord(name="baseline", config_hash="a"))
        cand_id = db.insert_experiment(ExperimentRecord(name="candidate", config_hash="b"))

        db.insert_benchmark(
            BenchmarkRecord(experiment_id=base_id, benchmark_name="math", score=0.9)
        )
        db.insert_benchmark(
            BenchmarkRecord(experiment_id=cand_id, benchmark_name="math", score=0.7)
        )

        result = db.detect_regression("math", "baseline", "candidate", threshold=0.05)
        assert result is not None
        assert result.regressed
        assert abs(result.delta - (-0.2)) < 1e-9
        db.close()

    def test_no_regression(self) -> None:
        db = _tmp_db()
        base_id = db.insert_experiment(ExperimentRecord(name="baseline", config_hash="a"))
        cand_id = db.insert_experiment(ExperimentRecord(name="candidate", config_hash="b"))

        db.insert_benchmark(
            BenchmarkRecord(experiment_id=base_id, benchmark_name="math", score=0.8)
        )
        db.insert_benchmark(
            BenchmarkRecord(experiment_id=cand_id, benchmark_name="math", score=0.85)
        )

        result = db.detect_regression("math", "baseline", "candidate")
        assert result is not None
        assert not result.regressed
        db.close()


class TestExperimentEnhancements:
    def test_count_experiments(self) -> None:
        db = _tmp_db()
        db.insert_experiment(ExperimentRecord(name="a", config_hash="1", status="running"))
        db.insert_experiment(ExperimentRecord(name="b", config_hash="2", status="running"))
        db.insert_experiment(ExperimentRecord(name="c", config_hash="3", status="done"))

        assert db.count_experiments() == 3
        assert db.count_experiments(status="running") == 2
        db.close()

    def test_delete_experiment(self) -> None:
        db = _tmp_db()
        exp_id = db.insert_experiment(ExperimentRecord(name="to-delete", config_hash="x"))
        db.insert_benchmark(
            BenchmarkRecord(experiment_id=exp_id, benchmark_name="test", score=0.5)
        )
        deleted = db.delete_experiment(exp_id)
        assert deleted
        assert db.get_experiment(exp_id) is None
        assert len(db.list_experiments()) == 0
        db.close()

    def test_experiment_lineage(self) -> None:
        db = _tmp_db()
        p1 = db.insert_experiment(ExperimentRecord(name="gen1", config_hash="1"))
        p2 = db.insert_experiment(ExperimentRecord(name="gen2", config_hash="2", parent_id=p1))
        p3 = db.insert_experiment(ExperimentRecord(name="gen3", config_hash="3", parent_id=p2))

        chain = db.get_experiment_lineage(p3)
        assert len(chain) == 3
        assert chain[0]["name"] == "gen1"
        assert chain[-1]["name"] == "gen3"
        db.close()

    def test_disk_usage(self) -> None:
        db = _tmp_db()
        db.insert_experiment(ExperimentRecord(name="x", config_hash="1"))
        usage = db.get_disk_usage()
        assert usage["size_bytes"] > 0
        assert usage["experiments"] == 1
        db.close()

    def test_best_experiment(self) -> None:
        db = _tmp_db()
        e1 = db.insert_experiment(ExperimentRecord(name="v1", config_hash="1"))
        e2 = db.insert_experiment(ExperimentRecord(name="v2", config_hash="2"))
        db.insert_benchmark(BenchmarkRecord(experiment_id=e1, benchmark_name="acc", score=0.8))
        db.insert_benchmark(BenchmarkRecord(experiment_id=e2, benchmark_name="acc", score=0.95))

        best = db.get_best_experiment("acc", higher_is_better=True)
        assert best is not None
        assert best["name"] == "v2"
        db.close()

    def test_get_latest_checkpoint(self) -> None:
        db = _tmp_db()
        exp_id = db.insert_experiment(ExperimentRecord(name="ckpt", config_hash="c"))
        db.insert_checkpoint(CheckpointRecord(experiment_id=exp_id, step=10, path="a.pt"))
        db.insert_checkpoint(CheckpointRecord(experiment_id=exp_id, step=20, path="b.pt"))

        latest = db.get_latest_checkpoint(exp_id)
        assert latest is not None
        assert latest["step"] == 20
        db.close()


class TestAuditLog:
    def test_audit_log(self) -> None:
        db = _tmp_db()
        db.insert_experiment(ExperimentRecord(name="audited", config_hash="z"))
        log = db.get_audit_log(entity="experiment")
        assert len(log) >= 1
        assert log[0]["entity"] == "experiment"
        db.close()
