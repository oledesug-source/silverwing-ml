"""Dynamic programming algorithms implemented from scratch."""


__all__ = [
    "fibonacci", "knapsack", "lcs", "edit_distance",
    "longest_palindrome", "coin_change", "matrix_chain_order",
    "rod_cutting", "subset_sum", "lis", "lis_nlogn",
]


def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using bottom-up DP.

    ``fibonacci(0) = 0``, ``fibonacci(1) = 1``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def knapsack(weights: list, values: list, capacity: int) -> tuple[int, list]:
    """0/1 Knapsack problem with backtracking to recover chosen items.

    Returns ``(max_value, list_of_chosen_indices)``.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                val = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                if val > dp[i][w]:
                    dp[i][w] = val

    chosen = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(i - 1)
            w -= weights[i - 1]
    chosen.reverse()

    return dp[n][capacity], chosen


def lcs(s1: str, s2: str) -> str:
    """Return the longest common subsequence of two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))


def edit_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    return dp[m][n]


def longest_palindrome(s: str) -> str:
    """Return the longest palindromic substring using expand-around-center."""
    if not s:
        return ""

    best_start = 0
    best_len = 1

    def _expand(left: int, right: int) -> tuple[int, int]:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - left - 1

    for i in range(len(s)):
        start1, len1 = _expand(i, i)
        if len1 > best_len:
            best_start, best_len = start1, len1
        start2, len2 = _expand(i, i + 1)
        if len2 > best_len:
            best_start, best_len = start2, len2

    return s[best_start : best_start + best_len]


def coin_change(coins: list, amount: int) -> int:
    """Return the minimum number of coins needed to make ``amount``.

    Returns -1 if the amount cannot be made.
    """
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a and dp[a - coin] + 1 < dp[a]:
                dp[a] = dp[a - coin] + 1

    return dp[amount] if dp[amount] != float("inf") else -1


def matrix_chain_order(dims: list) -> tuple[int, list]:
    """Matrix chain multiplication — optimal parenthesization.

    ``dims`` is a list of dimensions ``[d0, d1, ..., dn]`` where matrix i
    has dimensions ``dims[i] x dims[i+1]``.
    Returns ``(min_scalar_ops, split_points)`` where ``split_points[i][j]``
    records the optimal split for chain ``i..j``.
    """
    n = len(dims) - 1
    if n <= 0:
        return 0, []

    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    return dp[0][n - 1], split


def rod_cutting(prices: list, n: int) -> int:
    """Rod cutting problem — maximum revenue from cutting a rod of length n.

    ``prices[i]`` is the price for a rod of length ``i+1``.
    """
    dp = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            dp[j] = max(dp[j], prices[i] + dp[j - i - 1])
    return dp[n]


def subset_sum(nums: list, target: int) -> bool:
    """Determine if any subset of ``nums`` sums to exactly ``target``."""
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]

    return dp[target]


def lis(arr: list) -> list:
    """Longest increasing subsequence in O(n^2) with backtracking."""
    if not arr:
        return []

    n = len(arr)
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    best_idx = 0
    for i in range(n):
        if dp[i] > dp[best_idx]:
            best_idx = i

    result = []
    idx = best_idx
    while idx != -1:
        result.append(arr[idx])
        idx = parent[idx]
    result.reverse()
    return result


def lis_nlogn(arr: list) -> list:
    """Longest increasing subsequence in O(n log n) using patience sorting.

    Returns the actual subsequence, not just its length.
    """
    if not arr:
        return []

    import bisect

    n = len(arr)
    tails = []
    tail_indices = []
    parent = [-1] * n

    for i, num in enumerate(arr):
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            tail_indices.append(i)
        else:
            tails[pos] = num
            tail_indices[pos] = i
        if pos > 0:
            parent[i] = tail_indices[pos - 1]

    result = []
    idx = tail_indices[-1] if tail_indices else -1
    while idx != -1:
        result.append(arr[idx])
        idx = parent[idx]
    result.reverse()
    return result
