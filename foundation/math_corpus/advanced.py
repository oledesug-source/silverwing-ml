"""Advanced problem generators (M18): linear algebra, advanced probability,
statistics.

Follows the M08 contract exactly: ``gen(rng) -> Problem`` where the answer
AND the step-by-step solution are computed in code, keeping the corpus
self-consistent, verifiable and reproducible under a seeded Random (M01).

The ``solution`` field doubles as the chain-of-thought supervision target,
so SFT on these problems teaches derivation-style answers rather than bare
results.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from fractions import Fraction

from .problems import Problem, _frac


def _vec_str(v: list[int]) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def _mat_str(m: list[list[int]]) -> str:
    rows = ["[" + " ".join(f"{x:>3}" for x in row) + "]" for row in m]
    return "\n        ".join(rows)


def _fmt_num(x: float) -> str:
    """Render a float that is (almost) integral as an int."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------

def _gen_linear_algebra(rng: random.Random) -> Problem:
    kind = rng.randrange(6)

    if kind == 0:  # dot product
        u = [rng.randint(-9, 9) for _ in range(3)]
        v = [rng.randint(-9, 9) for _ in range(3)]
        dot = sum(a * b for a, b in zip(u, v))
        terms = " + ".join(f"({a})*({b})" for a, b in zip(u, v))
        return Problem(
            f"Compute the dot product u · v for u = {_vec_str(u)} and v = {_vec_str(v)}.",
            str(dot),
            f"u · v = {terms} = {dot}.",
        )

    if kind == 1:  # euclidean norm (pythagorean triples scaled)
        base = rng.choice([(3, 4), (6, 8), (5, 12), (8, 15), (9, 12), (7, 24)])
        scale = rng.randint(1, 3)
        u = [base[0] * scale, base[1] * scale]
        norm2 = u[0] ** 2 + u[1] ** 2
        norm = int(math.isqrt(norm2))
        return Problem(
            f"Find the Euclidean norm ||v|| for v = {_vec_str([u[0], u[1], 0])}.",
            str(norm),
            f"||v||^2 = ({u[0]})^2 + ({u[1]})^2 + 0^2 = {norm2}, so ||v|| = sqrt({norm2}) = {norm}.",
        )

    if kind == 2:  # 2x2 determinant
        m = [[rng.randint(-6, 6) for _ in range(2)] for _ in range(2)]
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        return Problem(
            f"Compute the determinant of the matrix\n{_mat_str(m)}",
            str(det),
            f"det = ({m[0][0]})*({m[1][1]}) - ({m[0][1]})*({m[1][0]}) "
            f"= {m[0][0] * m[1][1]} - {m[0][1] * m[1][0]} = {det}.",
        )

    if kind == 3:  # 2x2 matrix product
        A = [[rng.randint(-4, 4) for _ in range(2)] for _ in range(2)]
        B = [[rng.randint(-4, 4) for _ in range(2)] for _ in range(2)]
        C = [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        detail = "; ".join(
            f"C[{i}][{j}] = ({A[i][0]})*({B[0][j]}) + ({A[i][1]})*({B[1][j]}) = {C[i][j]}"
            for i in range(2)
            for j in range(2)
        )
        return Problem(
            f"Multiply the matrices A =\n{_mat_str(A)}\nand B =\n{_mat_str(B)}",
            "\n" + _mat_str(C),
            "Row-by-column products: " + detail + ".",
        )

    if kind == 4:  # 2x2 linear system via Cramer's rule (integer solution by design)
        p, q = rng.randint(-5, 5), rng.randint(-5, 5)
        while p == 0 and q == 0:
            p, q = rng.randint(-5, 5), rng.randint(-5, 5)
        a, b = rng.randint(1, 5), rng.randint(-4, 4)
        c, d = rng.randint(-4, 4), rng.randint(1, 5)
        while a * d - b * c == 0:
            a, b = rng.randint(1, 5), rng.randint(-4, 4)
            c, d = rng.randint(-4, 4), rng.randint(1, 5)
        e, f = a * p + b * q, c * p + d * q
        det = a * d - b * c
        det_x = e * d - b * f  # equals p * det by construction
        det_y = a * f - e * c  # equals q * det by construction
        return Problem(
            f"Solve the system: {a}x {'+' if b >= 0 else '-'} {abs(b)}y = {e}; "
            f"{c}x {'+' if d >= 0 else '-'} {abs(d)}y = {f}.",
            f"x = {p}, y = {q}",
            f"Cramer's rule: det = ({a})*({d}) - ({b})*({c}) = {det}; "
            f"det_x = ({e})*({d}) - ({b})*({f}) = {det_x}; "
            f"det_y = ({a})*({f}) - ({e})*({c}) = {det_y}. "
            f"So x = {det_x}/{det} = {p} and y = {det_y}/{det} = {q}.",
        )

    # kind == 5: eigenvalues of an upper-triangular 2x2 (diagonal entries)
    r1, r2 = rng.randint(-6, 6), rng.randint(-6, 6)
    while r1 == r2 or r1 == 0 or r2 == 0:
        r1, r2 = rng.randint(-6, 6), rng.randint(-6, 6)
    b01 = rng.choice([-4, -2, -1, 1, 2, 3])
    m = [[min(r1, r2), b01], [0, max(r1, r2)]]
    lo, hi = min(r1, r2), max(r1, r2)
    return Problem(
        f"Find the eigenvalues of the matrix\n{_mat_str(m)}",
        f"{lo} and {hi}",
        f"The matrix is upper triangular, and the eigenvalues of a triangular matrix "
        f"are exactly its diagonal entries. Therefore the eigenvalues are "
        f"λ = {lo} and λ = {hi}. (Check: trace = {lo + hi}, det = {lo * hi}, matching "
        f"(λ - ({lo}))(λ - ({hi})).)",
    )


# ---------------------------------------------------------------------------
# advanced probability
# ---------------------------------------------------------------------------

def _gen_advanced_probability(rng: random.Random) -> Problem:
    kind = rng.randrange(6)

    if kind == 0:  # combinations
        n = rng.randint(5, 10)
        k = rng.randint(2, min(4, n - 1))
        val = math.comb(n, k)
        return Problem(
            f"In how many ways can a committee of {k} people be chosen from a group of {n} people?",
            str(val),
            f"This is 'n choose k': C({n}, {k}) = {n}!/({k}!*{n - k}!) = {val}.",
        )

    if kind == 1:  # permutations
        n = rng.randint(4, 8)
        k = rng.randint(2, min(3, n - 1))
        val = math.perm(n, k)
        return Problem(
            f"How many distinct ordered arrangements of {k} items can be made from {n} distinct items?",
            str(val),
            f"Order matters, so this is P({n}, {k}) = {n}!/({n}-{k})! = {val}.",
        )

    if kind == 2:  # conditional probability without replacement
        blue, red, green = rng.randint(3, 8), rng.randint(2, 7), rng.randint(2, 6)
        total = blue + red + green
        draw = rng.choice(["blue", "red", "green"])
        fav = {"blue": blue, "red": red, "green": green}[draw]
        second = fav - 1
        joint = Fraction(fav * second, total * (total - 1))
        return Problem(
            f"A bag holds {blue} blue, {red} red and {green} green marbles. One marble "
            f"is drawn and kept out, then a second is drawn. What is the probability "
            f"BOTH draws are {draw}? Give the fraction in lowest terms.",
            _frac(joint.numerator, joint.denominator),
            f"P(first {draw}) = {fav}/{total}. Without replacement P(second {draw}) = "
            f"{second}/{total - 1}. Multiply: ({fav}/{total})*({second}/{total - 1}) = "
            f"{joint.numerator}/{joint.denominator}.",
        )

    if kind == 3:  # Bayes' theorem (10% prevalence keeps integers tractable)
        pop = rng.choice([100, 200])
        sick = pop // 10
        healthy = pop - sick
        tp = sick * 9 // 10          # sensitivity 9/10
        fp = healthy // 20           # false-positive rate 1/20
        post = Fraction(tp, tp + fp)
        return Problem(
            f"A disease affects {sick} in every {pop} people. A test detects the disease "
            f"in 9 out of 10 sick people, and wrongly flags 1 in 20 healthy people. If a "
            f"random person tests positive, what is the probability they actually have "
            f"the disease? Lowest terms.",
            f"{post.numerator}/{post.denominator}",
            f"True positives: {sick}*9/10 = {tp}. False positives: {healthy}*1/20 = {fp}. "
            f"P(sick | positive) = {tp}/({tp}+{fp}) = {post.numerator}/{post.denominator}.",
        )

    if kind == 4:  # expected value with a constant shift
        faces = rng.choice([6, 8])
        bonus = rng.randint(0, 3)
        mean = (faces + 1) / 2 + bonus
        return Problem(
            f"A fair {faces}-sided die shows 1..{faces} and you win that many points plus "
            f"a flat {bonus}-point bonus each roll. What is the expected value of one roll?",
            _fmt_num(mean),
            f"E[die] = (1+...+{faces})/{faces} = {faces * (faces + 1) // 2}/{faces} = "
            f"{(faces + 1) / 2}. Linearity of expectation adds the constant: "
            f"E = {(faces + 1) / 2} + {bonus} = {_fmt_num(mean)}.",
        )

    # union of independent events via inclusion-exclusion
    pa = Fraction(rng.randint(1, 4), 10)
    pb = Fraction(rng.randint(1, 4), 10)
    both = pa * pb
    union = pa + pb - both
    return Problem(
        f"Events A and B are independent with P(A) = {pa} and P(B) = {pb}. "
        f"What is P(A or B)? Exact fraction.",
        f"{union.numerator}/{union.denominator}",
        f"Independence gives P(A and B) = {pa}*{pb} = {both}. Inclusion-exclusion: "
        f"P(A or B) = {pa} + {pb} - {both} = {union.numerator}/{union.denominator}.",
    )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _gen_statistics(rng: random.Random) -> Problem:
    kind = rng.randrange(3)

    if kind == 0:  # mean over an integer-result dataset
        n = rng.choice([5, 6])
        mean = rng.randint(6, 15)
        dev = rng.sample(range(-4, 5), n)
        data = [mean + d for d in dev]
        total = sum(data)
        ds = ", ".join(map(str, data))
        return Problem(
            f"Find the mean of the dataset: {ds}.",
            str(mean),
            f"Sum = {' + '.join(map(str, data))} = {total}; count = {n}; "
            f"mean = {total}/{n} = {mean}.",
        )

    if kind == 1:  # median
        data = [rng.randint(1, 20) for _ in range(rng.choice([5, 6]))]
        s = sorted(data)
        n = len(s)
        med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        med_s = _fmt_num(float(med)) if isinstance(med, float) else str(med)
        ds = ", ".join(map(str, data))
        return Problem(
            f"Find the median of the dataset: {ds}.",
            med_s,
            f"Sorted: {', '.join(map(str, s))}. With n = {n}, the median is "
            + (
                f"the middle value {s[n // 2]}."
                if n % 2
                else f"the average of the two middle values: ({s[n // 2 - 1]} + {s[n // 2]})/2 = {med_s}."
            ),
        )

    # population variance from mean-deviation form (clean numbers)
    mu = rng.randint(5, 12)
    dev = rng.choice([(-2, -1, 0, 1, 2), (-3, -1, 1, 3)])
    data = [mu + d for d in dev]
    sq = [d * d for d in dev]
    var = Fraction(sum(sq), len(dev))
    sd = math.sqrt(var)
    ds = ", ".join(map(str, data))
    return Problem(
        f"Compute the population variance of: {ds}.",
        f"{var.numerator}/{var.denominator}" if var.denominator != 1 else str(var.numerator),
        f"Mean μ = {mu}. Squared deviations from μ: "
        f"{', '.join(f'({d})^2 = {d * d}' for d in dev)}. Variance = "
        f"{sum(sq)}/{len(dev)} = {var} (σ = sqrt({var}) ≈ {_fmt_num(sd)}).",
    )


ADVANCED_GENERATORS: Mapping[str, Callable[[random.Random], Problem]] = {
    "linear_algebra": _gen_linear_algebra,
    "advanced_probability": _gen_advanced_probability,
    "statistics": _gen_statistics,
}

ADVANCED_REFERENCES: Mapping[str, object] = {
    "linear_algebra": {
        "title": "Linear Algebra",
        "definition": (
            "Linear algebra studies vectors, matrices and linear transformations. "
            "Vectors have magnitude and direction; matrices encode linear maps."
        ),
        "rules": (
            "Dot product: (u1,u2,u3)·(v1,v2,v3) = u1v1+u2v2+u3v3. "
            "Norm: ||v|| = sqrt(v·v). 2x2 determinant: ad - bc. "
            "Eigenvalues solve det(A - λI) = 0; for triangular matrices they are the diagonal entries. "
            "Cramer's rule solves Ax = b with x_i = det(A_i)/det(A)."
        ),
    },
    "advanced_probability": {
        "title": "Advanced Probability",
        "definition": (
            "Probability quantifies uncertainty on a 0-to-1 scale, extending basic "
            "counting to conditional events, independence and expectation."
        ),
        "rules": (
            "Combinations C(n,k)=n!/(k!(n-k)!); permutations P(n,k)=n!/(n-k)!. "
            "Without replacement multiply updated fractions. Bayes: P(A|B) relates prior "
            "and likelihood. E[X+a] = E[X]+a (linearity). Inclusion-exclusion: "
            "P(A∪B) = P(A)+P(B)-P(A∩B), with P(A∩B)=P(A)P(B) when independent."
        ),
    },
    "statistics": {
        "title": "Statistics",
        "definition": (
            "Statistics summarizes data through measures of center (mean, median) "
            "and spread (variance, standard deviation)."
        ),
        "rules": (
            "Mean = sum/count. Median = middle value of the sorted data (or average of "
            "the two middle values). Population variance = Σ(x-μ)^2/n; σ = sqrt(variance)."
        ),
    },
}
