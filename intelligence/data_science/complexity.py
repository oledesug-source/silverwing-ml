"""Complexity analysis utilities for measuring time and space usage."""

import time
import tracemalloc

__all__ = [
    "measure_complexity", "amortized_analysis", "space_complexity",
]


def measure_complexity(func, *args, repeats: int = 10, **kwargs) -> dict:
    """Measure the average execution time and peak memory of ``func``.

    Runs ``func(*args, **kwargs)`` ``repeats`` times and reports
    the average time (seconds) and peak memory delta (bytes).
    """
    tracemalloc.start()
    _, peak_before = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times = []
    peak_mems = []

    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peak_mems.append(peak - peak_before if peak > peak_before else 0)

    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "avg_memory": sum(peak_mems) / len(peak_mems),
        "repeats": repeats,
    }


def amortized_analysis() -> dict:
    """Demonstrate amortised cost of a dynamic-array append sequence.

    Simulates appending ``n`` elements to a plain Python list and
    tracks the number of resize operations together with the
    per-element amortised cost.
    """
    import random

    random.seed(42)
    n = 1000
    data = []
    resize_count = 0
    total_copies = 0
    costs = []

    for _i in range(n):
        if len(data) > 0 and (len(data) & (len(data) - 1)) == 0 and len(data) > 1:
            resize_count += 1
            total_copies += len(data)

        old_len = len(data)
        data.append(random.random())
        cost = 1 if len(data) <= old_len + 1 else len(data)
        costs.append(cost)

    return {
        "n": n,
        "resizes": resize_count,
        "total_copies": total_copies,
        "amortized_cost_per_op": sum(costs) / n,
    }


def space_complexity(func) -> dict:
    """Approximate the space (memory) used by ``func()``.

    Calls ``func()`` once and reports the peak memory delta (bytes)
    measured via :mod:`tracemalloc`.
    """
    tracemalloc.start()
    peak_before = tracemalloc.get_traced_memory()[1]
    result = func()
    _, peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_memory_bytes": max(0, peak_after - peak_before),
        "returned_type": type(result).__name__,
    }
