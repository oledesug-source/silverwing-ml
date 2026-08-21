"""Sorting algorithms implemented from scratch."""

import random
import time
from dataclasses import dataclass, field

__all__ = [
    "quicksort", "mergesort", "heapsort", "tim_sort_run", "tim_sort",
    "counting_sort", "radix_sort", "bucket_sort", "SortResult",
    "get_sort_stats",
]


@dataclass
class SortResult:
    """Result of a sorting algorithm benchmark."""

    sorted_list: list[float] = field(default_factory=list)
    comparisons: int = 0
    swaps: int = 0


def _median_of_three(arr: list, lo: int, hi: int) -> int:
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]:
        arr[mid], arr[hi] = arr[hi], arr[mid]
    return mid


def quicksort(arr: list) -> list:
    """In-place quicksort with median-of-3 pivot selection.

    Returns a new sorted list without modifying the input.
    """
    result = list(arr)
    if len(result) <= 1:
        return result

    stats = {"comparisons": 0, "swaps": 0}

    def _quicksort(a: list, lo: int, hi: int):
        if lo >= hi:
            return
        pivot_idx = _median_of_three(a, lo, hi)
        a[pivot_idx], a[hi] = a[hi], a[pivot_idx]
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            stats["comparisons"] += 1
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                stats["swaps"] += 1
                i += 1
        a[i], a[hi] = a[hi], a[i]
        stats["swaps"] += 1
        _quicksort(a, lo, i - 1)
        _quicksort(a, i + 1, hi)

    _quicksort(result, 0, len(result) - 1)
    return result


def mergesort(arr: list) -> list:
    """Bottom-up iterative mergesort.

    Returns a new sorted list without modifying the input.
    """
    result = list(arr)
    n = len(result)
    if n <= 1:
        return result

    comparisons = 0
    width = 1
    while width < n:
        for i in range(0, n, width * 2):
            left = i
            mid = min(i + width, n)
            right = min(i + width * 2, n)
            merged = []
            l, r = left, mid
            while l < mid and r < right:
                comparisons += 1
                if result[l] <= result[r]:
                    merged.append(result[l])
                    l += 1
                else:
                    merged.append(result[r])
                    r += 1
            while l < mid:
                merged.append(result[l])
                l += 1
            while r < right:
                merged.append(result[r])
                r += 1
            for j in range(len(merged)):
                result[i + j] = merged[j]
        width *= 2
    return result


def heapsort(arr: list) -> list:
    """In-place heapsort.

    Returns a new sorted list without modifying the input.
    """
    result = list(arr)
    n = len(result)
    if n <= 1:
        return result

    comparisons = 0
    swaps = 0

    def _sift_down(a, start, end):
        nonlocal comparisons, swaps
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end:
                comparisons += 1
                if a[child] < a[child + 1]:
                    child += 1
            comparisons += 1
            if a[root] < a[child]:
                a[root], a[child] = a[child], a[root]
                swaps += 1
                root = child
            else:
                break

    for i in range(n // 2 - 1, -1, -1):
        _sift_down(result, i, n - 1)

    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        swaps += 1
        _sift_down(result, 0, i - 1)

    return result


def tim_sort_run(arr: list, start: int, end: int) -> int:
    """Detect a natural run in the array starting at ``start``.

    Extends to ``end`` and returns the length of the run.
    If the run is strictly descending it is reversed in-place.
    """
    if start >= end:
        return 0

    run_end = start + 1
    if run_end < end and arr[run_end] < arr[start]:
        while run_end < end and arr[run_end] < arr[run_end - 1]:
            run_end += 1
        arr[start:run_end] = reversed(arr[start:run_end])
    else:
        while run_end < end and arr[run_end] >= arr[run_end - 1]:
            run_end += 1

    return run_end - start


def tim_sort(arr: list) -> list:
    """Simplified timsort: detect natural runs and merge them.

    Uses ``tim_sort_run`` for run detection then merges runs
    of increasing size similar to the standard timsort merge policy.
    """
    result = list(arr)
    n = len(result)
    if n <= 1:
        return result

    runs = []
    i = 0
    while i < n:
        run_len = tim_sort_run(result, i, n)
        runs.append((i, run_len))
        i += run_len

    while len(runs) > 1:
        new_runs = []
        j = 0
        while j < len(runs):
            if j + 1 < len(runs):
                start1, len1 = runs[j]
                start2, len2 = runs[j + 1]
                merged = []
                left = start1
                right = start2
                end1 = start1 + len1
                end2 = start2 + len2
                while left < end1 and right < end2:
                    if result[left] <= result[right]:
                        merged.append(result[left])
                        left += 1
                    else:
                        merged.append(result[right])
                        right += 1
                while left < end1:
                    merged.append(result[left])
                    left += 1
                while right < end2:
                    merged.append(result[right])
                    right += 1
                for k in range(len(merged)):
                    result[start1 + k] = merged[k]
                new_runs.append((start1, len1 + len2))
                j += 2
            else:
                new_runs.append(runs[j])
                j += 1
        runs = new_runs

    return result


def counting_sort(arr: list, max_val: int) -> list:
    """Integer counting sort for values in [0, max_val].

    Returns a new sorted list.
    """
    if not arr:
        return []

    count = [0] * (max_val + 1)
    for v in arr:
        count[v] += 1

    result = []
    for i, c in enumerate(count):
        result.extend([i] * c)
    return result


def _get_digit(number: int, d: int) -> int:
    return (number // (10 ** d)) % 10


def radix_sort(arr: list) -> list:
    """LSD radix sort for non-negative integers.

    Returns a new sorted list.
    """
    if not arr:
        return []

    result = list(arr)
    max_val = max(result)
    d = 0
    while max_val // (10 ** d) > 0:
        buckets = [[] for _ in range(10)]
        for num in result:
            buckets[_get_digit(num, d)].append(num)
        result = []
        for bucket in buckets:
            result.extend(bucket)
        d += 1
    return result


def bucket_sort(arr: list) -> list:
    """Bucket sort for floats in [0, 1).

    Returns a new sorted list.
    """
    if not arr:
        return []

    n = len(arr)
    buckets = [[] for _ in range(n)]
    for v in arr:
        idx = min(int(v * n), n - 1)
        buckets[idx].append(v)

    result = []
    for bucket in buckets:
        bucket.sort()
        result.extend(bucket)
    return result


def get_sort_stats() -> dict:
    """Benchmark all sorting algorithms on the same random array.

    Returns a dict mapping algorithm name to ``SortResult``.
    """
    random.seed(42)
    data = [random.random() * 1000 for _ in range(500)]
    results = {}

    for name, func in [
        ("quicksort", quicksort),
        ("mergesort", mergesort),
        ("heapsort", heapsort),
        ("tim_sort", tim_sort),
    ]:
        arr = list(data)
        t0 = time.perf_counter()
        sorted_arr = func(arr)
        time.perf_counter() - t0
        results[name] = SortResult(sorted_list=sorted_arr, comparisons=0, swaps=0)

    arr = list(data)
    t0 = time.perf_counter()
    sorted_arr = bucket_sort(arr)
    time.perf_counter() - t0
    results["bucket_sort"] = SortResult(sorted_list=sorted_arr, comparisons=0, swaps=0)

    return results
