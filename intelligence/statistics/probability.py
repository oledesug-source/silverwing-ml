"""Probability theory primitives: events, combinatorics, Markov chains, random walks, and Monte Carlo methods."""

from __future__ import annotations

import random

__all__ = [
    "ProbabilitySpace",
    "Combinatorics",
    "MarkovChain",
    "RandomWalk",
    "MonteCarlo",
]


class ProbabilitySpace:
    """A finite sample space with add_event, probability, conditional_probability, and bayes_theorem.

    Events are stored as frozensets of outcomes.
    """

    def __init__(self) -> None:
        self._outcomes: list = []
        self._outcome_probs: dict = {}
        self._events: dict[str, frozenset] = {}
        self._event_probs: dict[str, float] = {}

    def add_outcome(self, outcome, probability: float = 0.0) -> None:
        """Register an outcome with its probability."""
        self._outcomes.append(outcome)
        self._outcome_probs[outcome] = probability

    def add_event(self, name: str, outcomes: set) -> None:
        """Register a named event as a set of outcomes."""
        self._events[name] = frozenset(outcomes)
        prob = sum(self._outcome_probs.get(o, 0.0) for o in outcomes)
        self._event_probs[name] = prob

    def probability(self, event_name: str) -> float:
        """Return the probability of a registered event."""
        return self._event_probs.get(event_name, 0.0)

    def conditional_probability(self, event_a: str, event_b: str) -> float:
        """Return P(A | B) = P(A ∩ B) / P(B)."""
        p_b = self.probability(event_b)
        if p_b == 0.0:
            return 0.0
        intersection = self._events[event_a] & self._events[event_b]
        p_intersection = sum(self._outcome_probs.get(o, 0.0) for o in intersection)
        return p_intersection / p_b

    def bayes_theorem(self, event_a: str, event_b: str) -> float:
        """Return P(A | B) using Bayes' theorem: P(B|A)*P(A) / P(B)."""
        self.probability(event_a)
        p_b = self.probability(event_b)
        if p_b == 0.0:
            return 0.0
        intersection = self._events[event_a] & self._events[event_b]
        p_a_and_b = sum(self._outcome_probs.get(o, 0.0) for o in intersection)
        return (p_a_and_b) / p_b


class Combinatorics:
    """Combinatorial functions: permutations, combinations, factorials, derangements, Catalan and Stirling numbers."""

    @staticmethod
    def factorial(n: int) -> int:
        """Return n!."""
        if n < 0:
            raise ValueError("factorial undefined for negative integers")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def permutation(self, n: int, k: int) -> int:
        """Return P(n, k) = n! / (n-k)!."""
        if k < 0 or k > n:
            raise ValueError("k must be between 0 and n")
        result = 1
        for i in range(n - k + 1, n + 1):
            result *= i
        return result

    def combination(self, n: int, k: int) -> int:
        """Return C(n, k) = n! / (k! * (n-k)!)."""
        if k < 0 or k > n:
            raise ValueError("k must be between 0 and n")
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(1, k + 1):
            result = result * (n - k + i) // i
        return result

    def derangement(self, n: int) -> int:
        """Return !n, the number of permutations of n elements with no fixed points."""
        if n == 0:
            return 1
        if n == 1:
            return 0
        d_prev2 = 1
        d_prev1 = 0
        for i in range(2, n + 1):
            d_current = (i - 1) * (d_prev1 + d_prev2)
            d_prev2 = d_prev1
            d_prev1 = d_current
        return d_prev1

    def catalan_number(self, n: int) -> int:
        """Return the n-th Catalan number C(2n,n)/(n+1)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.combination(2 * n, n) // (n + 1)

    def stirling_number_second(self, n: int, k: int) -> int:
        """Return the Stirling number of the second kind S(n, k)."""
        if k < 0 or k > n:
            return 0
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        table = [[0] * (k + 1) for _ in range(n + 1)]
        table[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
        return table[n][k]


class MarkovChain:
    """A discrete-time Markov chain with transition matrix, steady state, simulation, and absorption probabilities."""

    def __init__(self, states: list[str], transition_matrix: list[list[float]]) -> None:
        """Initialise with a list of state names and a row-stochastic transition matrix."""
        self.states = list(states)
        self.n = len(self.states)
        self._state_index = {s: i for i, s in enumerate(self.states)}
        self.transition_matrix = [list(row) for row in transition_matrix]

    @property
    def steady_state(self) -> dict[str, float]:
        """Compute the steady-state distribution by power iteration."""
        vec = [1.0 / self.n] * self.n
        for _ in range(10000):
            new_vec = [0.0] * self.n
            for j in range(self.n):
                for i in range(self.n):
                    new_vec[j] += vec[i] * self.transition_matrix[i][j]
            if all(abs(new_vec[j] - vec[j]) < 1e-12 for j in range(self.n)):
                vec = new_vec
                break
            vec = new_vec
        return {self.states[i]: vec[i] for i in range(self.n)}

    def step(self, current_state: str, rng: random.Random | None = None) -> str:
        """Take one transition from the given state."""
        r = rng or random
        idx = self._state_index[current_state]
        probs = self.transition_matrix[idx]
        cumsum = 0.0
        roll = r.random()
        for j, p in enumerate(probs):
            cumsum += p
            if roll < cumsum:
                return self.states[j]
        return self.states[-1]

    def simulate(self, start: str, steps: int, rng: random.Random | None = None) -> list[str]:
        """Simulate a path of length *steps* beginning at *start*."""
        path = [start]
        current = start
        for _ in range(steps):
            current = self.step(current, rng)
            path.append(current)
        return path

    def absorption_probabilities(self, absorbing: list[str]) -> dict[str, dict[str, float]]:
        """Compute absorption probabilities for transient states into absorbing states.

        Only valid for absorbing Markov chains.
        """
        abs_idx = [self._state_index[a] for a in absorbing]
        trans_idx = [i for i in range(self.n) if i not in abs_idx]
        if not trans_idx:
            return {a: {a: 1.0} for a in absorbing}

        def _idx(i: int) -> int:
            return trans_idx.index(i)

        size = len(trans_idx)
        Q = [[0.0] * size for _ in range(size)]
        R = [[0.0] * len(abs_idx) for _ in range(size)]
        for i_pos, i in enumerate(trans_idx):
            for j_pos, j in enumerate(trans_idx):
                Q[i_pos][j_pos] = self.transition_matrix[i][j]
            for a_pos, a in enumerate(abs_idx):
                R[i_pos][a_pos] = self.transition_matrix[i][a]

        I_mat = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        N = _matrix_inverse(_matrix_subtract(I_mat, Q))
        B = _matrix_multiply(N, R)

        result: dict[str, dict[str, float]] = {}
        for i_pos, i in enumerate(trans_idx):
            result[self.states[i]] = {absorbing[a]: B[i_pos][a] for a in range(len(abs_idx))}
        for a in absorbing:
            result[a] = {ab: 1.0 if ab == a else 0.0 for ab in absorbing}
        return result


class RandomWalk:
    """Discrete random walk in 1-D and 2-D with step, position, and hitting time."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._x = 0
        self._y = 0

    @property
    def position(self) -> tuple[int, int]:
        """Return current (x, y) position."""
        return (self._x, self._y)

    def step(self, directions: str = "udlr") -> tuple[int, int]:
        """Take a single step in a random direction from *directions* string."""
        d = self._rng.choice(list(directions))
        if d == "u":
            self._y += 1
        elif d == "d":
            self._y -= 1
        elif d == "l":
            self._x -= 1
        elif d == "r":
            self._x += 1
        return self.position

    @staticmethod
    def simulate_1d(steps: int, p: float = 0.5, rng: random.Random | None = None) -> list[int]:
        """Return the full path of a 1-D walk of length *steps* starting at 0."""
        r = rng or random.Random()
        path = [0]
        pos = 0
        for _ in range(steps):
            pos += 1 if r.random() < p else -1
            path.append(pos)
        return path

    @staticmethod
    def simulate_2d(steps: int, rng: random.Random | None = None) -> list[tuple[int, int]]:
        """Return the full path of a 2-D walk of length *steps* starting at (0, 0)."""
        r = rng or random.Random()
        x, y = 0, 0
        path = [(0, 0)]
        for _ in range(steps):
            d = r.choice(["u", "d", "l", "r"])
            if d == "u":
                y += 1
            elif d == "d":
                y -= 1
            elif d == "l":
                x -= 1
            else:
                x += 1
            path.append((x, y))
        return path

    @staticmethod
    def hitting_time(start: int, target: int, p: float = 0.5, max_steps: int = 10000, rng: random.Random | None = None) -> int:
        """Return the number of steps to first reach *target* from *start* in 1-D.

        Returns max_steps if the target is not reached.
        """
        r = rng or random.Random()
        pos = start
        for t in range(1, max_steps + 1):
            pos += 1 if r.random() < p else -1
            if pos == target:
                return t
        return max_steps


class MonteCarlo:
    """Monte Carlo estimation methods for π, integrals, and general integration."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def estimate_pi(self, n_samples: int = 100_000) -> float:
        """Estimate π using random points in the unit square / quarter circle."""
        inside = 0
        for _ in range(n_samples):
            x = self._rng.random()
            y = self._rng.random()
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / n_samples

    def estimate_integral(self, func, a: float, b: float, n_samples: int = 100_000) -> float:
        """Estimate ∫_a^b f(x) dx via the mean-value Monte Carlo formula."""
        total = 0.0
        for _ in range(n_samples):
            x = a + (b - a) * self._rng.random()
            total += func(x)
        return (b - a) * total / n_samples

    def monte_carlo_integration(self, func, bounds: list[tuple[float, float]], n_samples: int = 100_000) -> float:
        """Estimate a multi-dimensional integral using Monte Carlo sampling.

        *bounds* is a list of (lo, hi) tuples, one per dimension.
        """
        volume = 1.0
        for lo, hi in bounds:
            volume *= hi - lo
        total = 0.0
        for _ in range(n_samples):
            point = [lo + (hi - lo) * self._rng.random() for lo, hi in bounds]
            total += func(point)
        return volume * total / n_samples


def _matrix_subtract(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    return [[a[i][j] - b[i][j] for j in range(n)] for i in range(n)]


def _matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _matrix_inverse(mat: list[list[float]]) -> list[list[float]]:
    n = len(mat)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(mat)]
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n:] for i in range(n)]
