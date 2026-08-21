"""Searching algorithms implemented from scratch."""


__all__ = [
    "binary_search", "binary_search_left", "binary_search_right",
    "interpolation_search", "exponential_search", "jump_search",
    "fibonacci_search", "kmp_search", "boyer_moore_search", "Trie",
]


def binary_search(arr: list, target: float) -> int:
    """Standard binary search returning the index of target or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def binary_search_left(arr: list, target: float) -> int:
    """Return the leftmost insertion point for target in the sorted array."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search_right(arr: list, target: float) -> int:
    """Return the rightmost insertion point for target in the sorted array."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def interpolation_search(arr: list, target: float) -> int:
    """Interpolation search on uniformly distributed sorted data.

    Returns the index of target or -1.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi and arr[lo] <= target <= arr[hi]:
        if lo == hi:
            return lo if arr[lo] == target else -1
        pos = lo + ((target - arr[lo]) * (hi - lo)) // (arr[hi] - arr[lo])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            lo = pos + 1
        else:
            hi = pos - 1
    return -1


def exponential_search(arr: list, target: float) -> int:
    """Exponential search — O(log n) on unbounded / sorted arrays.

    Returns the index of target or -1.
    """
    if not arr:
        return -1
    if arr[0] == target:
        return 0

    bound = 1
    while bound < len(arr) and arr[bound] <= target:
        bound *= 2

    lo = bound // 2
    hi = min(bound, len(arr) - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def jump_search(arr: list, target: float) -> int:
    """Jump search — O(sqrt(n)) on sorted arrays.

    Returns the index of target or -1.
    """
    n = len(arr)
    if n == 0:
        return -1

    import math
    step = int(math.sqrt(n))
    prev = 0

    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1

    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1


def fibonacci_search(arr: list, target: float) -> int:
    """Fibonacci search on sorted arrays.

    Returns the index of target or -1.
    """
    n = len(arr)
    if n == 0:
        return -1

    fib2 = 0
    fib1 = 1
    fib = fib1 + fib2
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2

    offset = -1
    while fib > 1:
        i = min(offset + fib2, n - 1)
        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            return i

    if fib1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1
    return -1


def kmp_search(text: str, pattern: str) -> list:
    """Knuth-Morris-Pratt string matching.

    Returns a list of all starting indices where pattern occurs in text.
    """
    if not pattern:
        return list(range(len(text) + 1))

    def _build_lps(p: str) -> list:
        lps = [0] * len(p)
        length = 0
        i = 1
        while i < len(p):
            if p[i] == p[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return lps

    lps = _build_lps(pattern)
    results = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            results.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j:
                j = lps[j - 1]
            else:
                i += 1
    return results


def boyer_moore_search(text: str, pattern: str) -> list:
    """Simplified Boyer-Moore string matching using bad-character heuristic.

    Returns a list of all starting indices where pattern occurs in text.
    """
    if not pattern:
        return list(range(len(text) + 1))

    m = len(pattern)
    n = len(text)
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    results = []
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            results.append(s)
            s += m - bad_char.get(text[s + m], -1) if s + m < n else 1
        else:
            bad = bad_char.get(text[s + j], -1)
            s += max(1, j - bad)
    return results


class TrieNode:
    """A single node in a Trie."""

    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict = {}
        self.is_end: bool = False


class Trie:
    """Prefix tree supporting insert, search, prefix check, delete, and autocomplete."""

    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the trie."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        """Return True if the exact word exists in the trie."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word in the trie starts with the given prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def delete(self, word: str) -> None:
        """Delete a word from the trie if present."""

        def _delete(node: TrieNode, w: str, depth: int) -> bool:
            if depth == len(w):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0
            ch = w[depth]
            if ch not in node.children:
                return False
            should_delete = _delete(node.children[ch], w, depth + 1)
            if should_delete:
                del node.children[ch]
                return not node.is_end and len(node.children) == 0
            return False

        _delete(self.root, word, 0)

    def autocomplete(self, prefix: str) -> list:
        """Return all words that start with the given prefix, sorted lexicographically."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        results = []

        def _dfs(n: TrieNode, path: list):
            if n.is_end:
                results.append("".join(path))
            for ch in sorted(n.children):
                path.append(ch)
                _dfs(n.children[ch], path)
                path.pop()

        _dfs(node, list(prefix))
        return sorted(results)
