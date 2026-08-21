"""Tests for enhanced intelligence.operating_systems — file I/O, CPU monitor, resource tracker."""

import threading
import time

from intelligence.operating_systems.cpu_monitor import CpuMonitor, ResourceLimits, ResourceTracker
from intelligence.operating_systems.file_io import FileCache, FileManager, SafeWriter
from intelligence.operating_systems.memory import MemoryManager, MemoryStats
from intelligence.operating_systems.process import ProcessManager
from intelligence.operating_systems.sync import Lock, Semaphore, WorkerPool


class TestProcessManager:
    def test_cpu_count(self) -> None:
        assert ProcessManager.get_cpu_count() >= 1

    def test_pid(self) -> None:
        assert ProcessManager.get_pid() > 0

    def test_submit_task(self) -> None:
        pm = ProcessManager(max_workers=2)
        pm.start()
        task_id = pm.submit(lambda: 42, task_name="test-task")
        info = pm.get_info(task_id)
        assert info is not None
        assert info.name == "test-task"
        pm.shutdown()


class TestMemoryManager:
    def test_snapshot(self) -> None:
        mm = MemoryManager()
        stats = mm.snapshot()
        assert isinstance(stats, MemoryStats)
        assert stats.rss_mb >= 0

    def test_track_context(self) -> None:
        mm = MemoryManager()
        with mm.track("test_block"):
            _ = list(range(1000))
        assert "test_block" in mm._tracking


class TestLock:
    def test_acquire_release(self) -> None:
        lock = Lock(name="test-lock")
        with lock.acquire(timeout=1.0):
            assert lock.locked

    def test_timeout(self) -> None:
        lock = Lock(name="test-lock")
        acquired = [False]

        def hold_lock() -> None:
            with lock.acquire(timeout=10.0):
                time.sleep(0.1)

        t = threading.Thread(target=hold_lock)
        t.start()
        time.sleep(0.05)
        try:
            with lock.acquire(timeout=0.01):
                pass
        except TimeoutError:
            acquired[0] = True
        t.join()
        assert acquired[0]


class TestSemaphore:
    def test_concurrent_access(self) -> None:
        sem = Semaphore(max_concurrent=2)
        assert sem.available == 2
        with sem:
            assert sem.available == 1
        assert sem.available == 2


class TestWorkerPool:
    def test_submit_task(self) -> None:
        results = []
        pool = WorkerPool(max_workers=2)
        pool.start()
        pool.submit(lambda: results.append(1))
        pool.submit(lambda: results.append(2))
        time.sleep(0.3)
        pool.shutdown()
        assert sorted(results) == [1, 2]


class TestFileCache:
    def test_read_write(self) -> None:
        cache = FileCache(max_size_mb=1, ttl_seconds=60)
        cache.write("/test/file", b"hello")
        assert cache.read("/test/file") == b"hello"

    def test_miss(self) -> None:
        cache = FileCache()
        assert cache.read("/nonexistent") is None

    def test_invalidate(self) -> None:
        cache = FileCache()
        cache.write("/test", b"data")
        assert cache.invalidate("/test")
        assert cache.read("/test") is None

    def test_stats(self) -> None:
        cache = FileCache()
        cache.write("/a", b"12345")
        stats = cache.stats
        assert stats["entries"] == 1
        assert stats["hits"] == 0

    def test_ttl_expiry(self) -> None:
        cache = FileCache(ttl_seconds=0.01)
        cache.write("/test", b"data")
        time.sleep(0.02)
        assert cache.read("/test") is None

    def test_lru_eviction(self) -> None:
        cache = FileCache(max_size_mb=0.0001, ttl_seconds=60)
        cache.write("/a", b"x" * 50000)
        cache.write("/b", b"y" * 50000)
        stats = cache.stats
        assert stats["entries"] <= 2


class TestSafeWriter:
    def test_atomic_write(self, tmp_path) -> None:
        target = tmp_path / "output.txt"
        with SafeWriter(target) as tmp:
            tmp.write_bytes(b"hello world")
        assert target.read_bytes() == b"hello world"

    def test_cleanup_on_error(self, tmp_path) -> None:
        target = tmp_path / "fail.txt"
        try:
            with SafeWriter(target) as tmp:
                tmp.write_bytes(b"partial")
                raise ValueError("boom")
        except ValueError:
            pass
        assert not target.exists()


class TestFileManager:
    def test_dir_stats(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_bytes(b"hello")
        (tmp_path / "b.txt").write_bytes(b"world!")
        fm = FileManager()
        stats = fm.dir_stats(tmp_path)
        assert stats["file_count"] == 2
        assert stats["total_size_bytes"] == 11

    def test_checksum_file(self, tmp_path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"test data")
        fm = FileManager()
        h = fm.checksum_file(f)
        assert len(h) == 64

    def test_stream_file(self, tmp_path) -> None:
        f = tmp_path / "stream.bin"
        f.write_bytes(b"x" * 1000)
        fm = FileManager()
        chunks = list(fm.stream_file(f, chunk_size=100))
        assert len(chunks) == 10

    def test_ensure_dir(self, tmp_path) -> None:
        fm = FileManager()
        p = fm.ensure_dir(tmp_path / "a" / "b" / "c")
        assert p.exists()

    def test_cache_integration(self, tmp_path) -> None:
        fm = FileManager(cache_size_mb=1)
        fm.cache.write("/key", b"value")
        assert fm.cache.read("/key") == b"value"


class TestCpuMonitor:
    def test_snapshot(self) -> None:
        monitor = CpuMonitor(interval=0.1)
        stats = monitor.snapshot()
        assert stats.cpu_count >= 1

    def test_start_stop(self) -> None:
        monitor = CpuMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        history = monitor.get_history()
        assert len(history) >= 1

    def test_average(self) -> None:
        monitor = CpuMonitor(interval=0.05)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        avg = monitor.get_average(last_n=5)
        assert avg >= 0


class TestResourceTracker:
    def test_runtime_limit(self) -> None:
        tracker = ResourceTracker(limits=ResourceLimits(max_runtime_seconds=0.1))
        tracker.start()
        time.sleep(0.2)
        assert tracker.should_stop()
        violations = tracker.get_violations()
        assert any("Runtime exceeded" in v for v in violations)
        tracker.stop()

    def test_no_violations_initially(self) -> None:
        tracker = ResourceTracker(limits=ResourceLimits(max_runtime_seconds=60))
        tracker.start()
        assert not tracker.should_stop()
        assert len(tracker.get_violations()) == 0
        tracker.stop()

    def test_elapsed_seconds(self) -> None:
        tracker = ResourceTracker(limits=ResourceLimits(max_runtime_seconds=60))
        tracker.start()
        time.sleep(0.05)
        elapsed = tracker.elapsed_seconds()
        assert elapsed >= 0.04
        tracker.stop()

    def test_elapsed_before_start(self) -> None:
        tracker = ResourceTracker()
        assert tracker.elapsed_seconds() == 0.0

    def test_default_limits(self) -> None:
        tracker = ResourceTracker()
        tracker.start()
        time.sleep(0.02)
        assert not tracker.should_stop()
        tracker.stop()

    def test_stop_and_restart(self) -> None:
        tracker = ResourceTracker(limits=ResourceLimits(max_runtime_seconds=0.05))
        tracker.start()
        time.sleep(0.1)
        tracker.stop()
        assert tracker.elapsed_seconds() > 0


class TestMemoryManagerExtended:
    def test_check_available(self) -> None:
        mm = MemoryManager()
        result = mm.check_available(0)
        assert isinstance(result, bool)

    def test_warn_if_high(self) -> None:
        mm = MemoryManager(warn_threshold=-1.0)
        result = mm.warn_if_high()
        assert result is not None

    def test_warn_if_low(self) -> None:
        mm = MemoryManager(warn_threshold=1.0)
        result = mm.warn_if_high()
        assert result is None

    def test_stats_properties(self) -> None:
        from intelligence.operating_systems.memory import MemoryStats
        stats = MemoryStats(rss_bytes=1024 * 1024, vms_bytes=2 * 1024 * 1024, available_bytes=3 * 1024 * 1024)
        assert stats.rss_mb == 1.0
        assert stats.vms_mb == 2.0
        assert stats.available_mb == 3.0

    def test_tracking_multiple_blocks(self) -> None:
        mm = MemoryManager()
        with mm.track("block_a"):
            _ = [0] * 100
        with mm.track("block_b"):
            _ = [0] * 200
        assert "block_a" in mm._tracking
        assert "block_b" in mm._tracking


class TestCpuMonitorExtended:
    def test_callback(self) -> None:
        samples = []
        monitor = CpuMonitor(interval=0.05)
        monitor.on_sample(lambda s: samples.append(s))
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        assert len(samples) >= 1

    def test_average_empty(self) -> None:
        monitor = CpuMonitor()
        assert monitor.get_average(last_n=10) == 0.0

    def test_history_empty(self) -> None:
        monitor = CpuMonitor()
        assert monitor.get_history() == []

    def test_snapshot_properties(self) -> None:
        monitor = CpuMonitor()
        stats = monitor.snapshot()
        assert stats.cpu_count >= 1
        assert stats.load_per_core >= 0.0

    def test_snapshot_load_per_core_zero_count(self) -> None:
        from intelligence.operating_systems.cpu_monitor import CpuStats
        stats = CpuStats(cpu_count=0, load_avg_1m=5.0)
        assert stats.load_per_core == 0.0


class TestProcessManagerExtended:
    def test_submit_before_start(self) -> None:
        pm = ProcessManager(max_workers=2)
        try:
            pm.submit(lambda: 42)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_list_tasks(self) -> None:
        pm = ProcessManager(max_workers=2)
        pm.start()
        pm.submit(lambda: 1, task_name="t1")
        pm.submit(lambda: 2, task_name="t2")
        tasks = pm.list_tasks()
        assert len(tasks) == 2
        pm.shutdown()

    def test_shutdown_idempotent(self) -> None:
        pm = ProcessManager(max_workers=1)
        pm.start()
        pm.shutdown()
        pm.shutdown()


class TestSemaphoreExtended:
    def test_concurrent_threads(self) -> None:
        sem = Semaphore(max_concurrent=2)
        active = [0]
        max_active = [0]
        lock = threading.Lock()

        def worker():
            with sem:
                with lock:
                    active[0] += 1
                    max_active[0] = max(max_active[0], active[0])
                time.sleep(0.05)
                with lock:
                    active[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert max_active[0] <= 2


class TestLockExtended:
    def test_reentrant(self) -> None:
        lock = Lock(name="reentrant")
        with lock.acquire():
            with lock.acquire():
                assert True

    def test_name(self) -> None:
        lock = Lock(name="my-lock")
        assert lock.name == "my-lock"
