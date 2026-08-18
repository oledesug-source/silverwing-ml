"""Problem generators for the math training corpus (M08).

Each topic exposes a generator ``gen(rng) -> Problem`` producing a single
mathematical problem whose *answer and step-by-step solution are computed
in code*, so the released corpus is self-consistent and verifiable.  The
generators are deterministic under a seeded ``random.Random``, which keeps
the corpus reproducible (M01 rule) from the same committed config + seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping


@dataclass(frozen=True)
class Problem:
    question: str
    answer: str
    solution: str


@dataclass(frozen=True)
class TopicReference:
    title: str
    definition: str
    rules: str


def _frac(num: int, den: int) -> str:
    g = math.gcd(abs(num), abs(den))
    num //= g
    den //= g
    if den == 1:
        return str(num)
    return f"{num}/{den}"


def _sign_str(coeff: int) -> str:
    return "-" if coeff < 0 else "+"


def _poly_str(coeffs: list[int], var: str = "x") -> str:
    """Render a polynomial like [2, -5, 3] as '2x^2 - 5x + 3'."""
    terms = []
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = n - 1 - i
        if power == 0:
            terms.append(str(c))
        elif power == 1:
            terms.append(f"{c if c != 1 else ''}{var}")
        else:
            terms.append(f"{c if c != 1 else ''}{var}^{power}")
    if not terms:
        return "0"
    out = terms[0]
    for t in terms[1:]:
        sep = "-" if t.startswith("-") else "+"
        out += f" {sep} {t.lstrip('-+')}"
    return out


def _gen_arithmetic(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        op, fn = rng.choice([("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)])
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        q = f"Compute {a} {op} {b}."
        return Problem(q, str(fn(a, b)), f"{a} {op} {b} = {fn(a, b)}.")
    if kind == 1:
        a, b, c, d = (rng.randint(1, 9) for _ in range(4))
        num = a * d + c * b
        den = b * d
        return Problem(
            f"Compute {a}/{b} + {c}/{d} and give the answer in lowest terms.",
            _frac(num, den),
            f"Using the common denominator {b}*{d} = {den}: "
            f"{a}/{b} + {c}/{d} = ({a}*{d} + {c}*{b})/{den} = {num}/{den} = {_frac(num, den)}.",
        )
    p, x = rng.randint(5, 75), rng.choice([50, 100, 200, 400, 800, 1000])
    val = int(round(p * x / 100))
    return Problem(
        f"What is {p}% of {x}?",
        str(val),
        f"{p}% of {x} = ({p}/100) * {x} = {p * x / 100:.0f}.",
    )


def _gen_algebra(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        a, c = rng.randint(1, 5), rng.randint(1, 5)
        b, d = rng.randint(-9, 9), rng.randint(-9, 9)
        q = f"Expand ({a}x {_sign_str(b)} {abs(b)}) * ({c}x {_sign_str(d)} {abs(d)})."
        coeffs = [a * c, a * d + b * c, b * d]
        return Problem(
            q,
            _poly_str(coeffs),
            f"Apply FOIL: {a}*{c}x^2 + ({a}*{d} + {b}*{c})x + {b}*{d} = {_poly_str(coeffs)}.",
        )
    if kind == 1:
        r1, r2 = rng.randint(-8, 8), rng.randint(-8, 8)
        while r2 == 0 or r1 == 0:
            r1, r2 = rng.randint(-8, 8), rng.randint(-8, 8)
        q = f"Factor x^2 {_sign_str(-(r1 + r2))} {abs(r1 + r2)}x {_sign_str(r1 * r2)} {abs(r1 * r2)}."
        return Problem(
            q,
            f"(x {_sign_str(-r1)} {abs(r1)})(x {_sign_str(-r2)} {abs(r2)})",
            f"Find two numbers that add to {r1 + r2} and multiply to {r1 * r2}: "
            f"{r1} and {r2}.  Therefore x^2 {_sign_str(-(r1 + r2))} {abs(r1 + r2)}x "
            f"{_sign_str(r1 * r2)} {abs(r1 * r2)} = (x {_sign_str(-r1)} {abs(r1)})(x {_sign_str(-r2)} {abs(r2)}).",
        )
    r1, r2 = rng.randint(-8, 8), rng.randint(-8, 8)
    while r2 == r1 or r2 == 0:
        r1, r2 = rng.randint(-8, 8), rng.randint(-8, 8)
    q = f"Solve x^2 {_sign_str(-(r1 + r2))} {abs(r1 + r2)}x {_sign_str(r1 * r2)} {abs(r1 * r2)} = 0."
    return Problem(
        q,
        f"x = {r1} or x = {r2}",
        f"The roots must add to {r1 + r2} and multiply to {r1 * r2}.  They are {r1} and {r2}, "
        f"so x = {r1} or x = {r2}.",
    )


def _gen_linear(rng: random.Random) -> Problem:
    a = rng.randint(1, 9)
    c = rng.randint(1, 9)
    while c == a:
        c = rng.randint(1, 9)
    x0 = rng.randint(-10, 10)
    b = rng.randint(-20, 20)
    d = b + x0 * (a - c)
    q = f"Solve for x: {a}x {_sign_str(b)} {abs(b)} = {c}x {_sign_str(d)} {abs(d)}."
    return Problem(
        q,
        f"x = {x0}",
        f"Subtract {c}x from both sides: ({a} - {c})x {_sign_str(b)} {abs(b)} = {d}.  "
        f"Then ({a} - {c})x = {d} - {b} = {x0 * (a - c)}, so x = {x0 * (a - c)}/{a - c} = {x0}.",
    )


def _gen_functions(rng: random.Random) -> Problem:
    kind = rng.randrange(2)
    a, b = rng.randint(1, 9), rng.randint(-20, 20)
    x0 = rng.randint(-10, 10)
    if kind == 0:
        q = f"Evaluate f(x) = {a}x {_sign_str(b)} {abs(b)} at x = {x0}."
        return Problem(q, str(a * x0 + b), f"f({x0}) = {a}*{x0} {_sign_str(b)} {abs(b)} = {a * x0 + b}.")
    c, d = rng.randint(1, 5), rng.randint(-9, 9)
    gv = c * x0 + d
    fv = a * gv + b
    q = f"Given f(x) = {a}x {_sign_str(b)} {abs(b)} and g(x) = {c}x {_sign_str(d)} {abs(d)}, compute f(g({x0}))."
    return Problem(
        q,
        str(fv),
        f"First g({x0}) = {c}*{x0} {_sign_str(d)} {abs(d)} = {gv}.  "
        f"Then f({gv}) = {a}*{gv} {_sign_str(b)} {abs(b)} = {fv}.",
    )


def _gen_differentiation(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        a, n = rng.randint(1, 6), rng.randint(2, 4)
        q = f"Find the derivative of f(x) = {a}x^{n}."
        return Problem(
            q,
            f"f'(x) = {a * n}x^{n - 1}",
            f"By the power rule, d/dx[{a}x^{n}] = {a}*{n}x^({n} - 1) = {a * n}x^{n - 1}.",
        )
    if kind == 1:
        a, b, c, d = (rng.randint(-5, 5) for _ in range(4))
        while a == 0:
            a = rng.randint(-5, 5)
        while c == 0:
            c = rng.randint(-5, 5)
        q = f"Find the derivative of f(x) = {_poly_str([a, b, c, d])}."
        deriv = [a * 3, b * 2, c]
        return Problem(
            q,
            f"f'(x) = {_poly_str(deriv)}",
            f"Differentiate term by term: d/dx[{a}x^3] = {a * 3}x^2, d/dx[{b}x^2] = {b * 2}x, "
            f"d/dx[{c}x] = {c}, so f'(x) = {_poly_str(deriv)}.",
        )
    a, k = rng.randint(1, 5), rng.randint(1, 4)
    q = f"Find the derivative of f(x) = {a}*sin({k}x)."
    return Problem(
        q,
        f"f'(x) = {a * k}*cos({k}x)",
        f"By the chain rule, d/dx[{a}*sin({k}x)] = {a}*{k}*cos({k}x) = {a * k}*cos({k}x).",
    )


def _gen_integration(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        a, n = rng.randint(1, 6), rng.randint(1, 4)
        q = f"Evaluate the indefinite integral ∫ {a}x^{n} dx."
        return Problem(
            q,
            f"{_frac(a, n + 1)}x^{n + 1} + C",
            f"∫ {a}x^{n} dx = {a}/({n} + 1) x^({n} + 1) + C = {_frac(a, n + 1)}x^{n + 1} + C.",
        )
    if kind == 1:
        p, q_ = sorted((rng.randint(0, 4), rng.randint(1, 6)))
        a = rng.randint(1, 4)
        f = Fraction(a * (q_ ** 3 - p ** 3), 3)
        q = f"Evaluate the definite integral ∫ from {p} to {q_} of {a}x^2 dx."
        return Problem(
            q,
            _frac(f.numerator, f.denominator) if f.denominator != 1 else str(f.numerator),
            f"∫ from {p} to {q_} of {a}x^2 dx = [{a}x^3/3] from {p} to {q_} = "
            f"{a}*{q_}^3/3 - {a}*{p}^3/3 = {_frac(f.numerator, f.denominator) if f.denominator != 1 else str(f.numerator)}.",
        )
    a, b, c = (rng.randint(-4, 4) for _ in range(3))
    while a == 0:
        a = rng.randint(-4, 4)
    q = f"Evaluate the indefinite integral ∫ ({_poly_str([a, b, c])}) dx."
    answer = _integral_str(a, b, c)
    steps = f"∫ {a}x^2 dx = {_frac(a, 3)}x^3, ∫ {b}x dx = {_frac(b, 2)}x^2, ∫ {c} dx = {c}x"
    return Problem(q, answer + " + C", f"{steps}; combine the terms: {answer} + C.")


def _integral_str(a: int, b: int, c: int) -> str:
    terms = []
    if a:
        terms.append(f"{_frac(a, 3)}x^3")
    if b:
        terms.append(f"{_frac(b, 2)}x^2")
    if c:
        terms.append(str(c))
    if not terms:
        return "0"
    out = terms[0]
    for t in terms[1:]:
        neg = t.startswith("-")
        out += f" {'-' if neg else '+'} {t.lstrip('-')}"
    return out


def _gen_geometry(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        w, h = rng.randint(2, 15), rng.randint(2, 15)
        q = f"A rectangle has width {w} and height {h}.  Find its area and perimeter."
        return Problem(
            q,
            f"area = {w * h}, perimeter = {2 * (w + h)}",
            f"Area = {w}*{h} = {w * h}; perimeter = 2*({w} + {h}) = {2 * (w + h)}.",
        )
    if kind == 1:
        b, h = rng.randint(2, 14), rng.randint(2, 14)
        while (b * h) % 2 != 0:
            h = rng.randint(2, 14)
        q = f"A triangle has base {b} and height {h}.  Find its area."
        return Problem(q, str(b * h // 2), f"Area = (1/2)*{b}*{h} = {b * h // 2}.")
    r = rng.randint(1, 9)
    q = f"A circle has radius {r}.  Find its area and circumference (leave answers in terms of π)."
    return Problem(
        q,
        f"area = {r * r}π, circumference = {2 * r}π",
        f"Area = πr^2 = π*{r}^2 = {r * r}π; circumference = 2πr = 2*π*{r} = {2 * r}π.",
    )


def _gen_probability(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        k = rng.randint(1, 6)
        q = f"A fair six-sided die is rolled once.  What is the probability of rolling exactly {k}?"
        return Problem(q, "1/6", f"There is 1 favourable outcome out of 6 equally likely outcomes, so P = 1/6.")
    if kind == 1:
        k = rng.randint(1, 3)
        q = f"A fair coin is flipped {k} times.  What is the probability of {k} heads in a row?"
        ans = _frac(1, 2 ** k)
        return Problem(q, ans, f"Each flip has probability 1/2, so P = (1/2)^{k} = {ans}.")
    n, k = rng.randint(2, 10), rng.randint(1, 2)
    q = f"From {n} distinct items, how many subsets of size {k} exist?"
    return Problem(
        q,
        str(math.comb(n, k)),
        f"The number of k-subsets of an n-set is C({n}, {k}) = {n}!/({k}!({n} - {k})!) = {math.comb(n, k)}.",
    )


def _gen_number_theory(rng: random.Random) -> Problem:
    kind = rng.randrange(4)
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    if kind == 0:
        q = f"Find gcd({a}, {b})."
        g = math.gcd(a, b)
        return Problem(
            q,
            str(g),
            f"The greatest common divisor is the largest integer dividing both {a} and {b}, "
            f"which is gcd({a}, {b}) = {g}.",
        )
    if kind == 1:
        q = f"Find lcm({a}, {b})."
        l = a * b // math.gcd(a, b)
        return Problem(q, str(l), f"lcm({a}, {b}) = ({a}*{b}) / gcd({a}, {b}) = {a * b}/{math.gcd(a, b)} = {l}.")
    if kind == 2:
        m = rng.randint(5, 15)
        q = f"Compute {a} mod {m}."
        return Problem(q, str(a % m), f"{a} divided by {m} leaves remainder {a % m}, so {a} mod {m} = {a % m}.")
    q = f"Is {a} a prime number?"
    is_prime = _is_prime(a)
    return Problem(q, "yes" if is_prime else "no", _prime_reason(a))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _prime_reason(n: int) -> str:
    if _is_prime(n):
        return f"{n} has no divisor greater than 1 and less than itself, so it is prime."
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return f"{n} is divisible by {i} ({n} = {i}*{n // i}), so it is not prime."


_TRIG_VALUES: dict[str, dict[str, str]] = {
    "30": {"sin": "1/2", "cos": "√3/2", "tan": "√3/3"},
    "45": {"sin": "√2/2", "cos": "√2/2", "tan": "1"},
    "60": {"sin": "√3/2", "cos": "1/2", "tan": "√3"},
    "90": {"sin": "1", "cos": "0", "tan": "undefined"},
}


def _gen_trigonometry(rng: random.Random) -> Problem:
    kind = rng.randrange(3)
    if kind == 0:
        angle = rng.choice(["30", "45", "60", "90"])
        func = rng.choice(["sin", "cos", "tan"])
        q = f"Find the exact value of {func}({angle}°)."
        return Problem(q, _TRIG_VALUES[angle][func], f"The standard exact value is {func}({angle}°) = {_TRIG_VALUES[angle][func]}.")
    angle = rng.choice(["30", "60"])
    k = int(angle) // 30
    # solve sin(x) = 1/2 or cos(x) = 1/2 within [0°, 360°]
    if angle == "30":
        sols = "x = 30° or x = 150°"
        steps = f"sin(x) = 1/2 in [0°, 360°].  sin is positive in quadrants I and II: x = 30° and x = 180° - 30° = 150°."
    else:
        sols = "x = 60° or x = 300°"
        steps = f"cos(x) = 1/2 in [0°, 360°].  cos is positive in quadrants I and IV: x = 60° and x = 360° - 60° = 300°."
    q = f"Solve sin(x) = 1/2 for x in [0°, 360°]." if angle == "30" else f"Solve cos(x) = 1/2 for x in [0°, 360°]."
    return Problem(q, sols, steps)


PROBLEM_GENERATORS: Mapping[str, Callable[[random.Random], Problem]] = {
    "arithmetic": _gen_arithmetic,
    "algebra": _gen_algebra,
    "linear_equations": _gen_linear,
    "functions": _gen_functions,
    "differentiation": _gen_differentiation,
    "integration": _gen_integration,
    "geometry": _gen_geometry,
    "probability": _gen_probability,
    "number_theory": _gen_number_theory,
    "trigonometry": _gen_trigonometry,
}

REFERENCES: Mapping[str, TopicReference] = {
    "arithmetic": TopicReference(
        title="Arithmetic",
        definition="Arithmetic is the study of numbers and the basic operations performed on them: addition, subtraction, multiplication and division.",
        rules="Add and subtract by aligning place values.  Multiply digit by digit and add partial products.  A fraction a/b is in lowest terms when gcd(a, b) = 1.  A percentage p% is the fraction p/100.",
    ),
    "algebra": TopicReference(
        title="Algebra",
        definition="Algebra generalizes arithmetic by using symbols, typically letters, to represent numbers that may be unknown or variable.",
        rules="The distributive law a(b + c) = ab + ac lets you expand products.  To factor x^2 + px + q, find two numbers that add to p and multiply to q.  A quadratic x^2 + px + q = 0 with roots r1 and r2 satisfies r1 + r2 = -p and r1*r2 = q.",
    ),
    "linear_equations": TopicReference(
        title="Linear Equations",
        definition="A linear equation is an equality that can be written as ax + b = 0 with a ≠ 0; its graph is a straight line.",
        rules="To solve ax + b = cx + d, move the variable terms to one side: (a - c)x = d - b, then divide by the coefficient of x, provided a ≠ c.",
    ),
    "functions": TopicReference(
        title="Functions",
        definition="A function f maps each input x to exactly one output f(x).  Composition f(g(x)) means applying g first, then f.",
        rules="Evaluate a linear function f(x) = ax + b by substituting the input.  For a composition f(g(x)), compute g(x) first and substitute that value into f.",
    ),
    "differentiation": TopicReference(
        title="Differentiation",
        definition="The derivative f'(x) measures the rate of change of f at x; it is the slope of the tangent line to the graph of f.",
        rules="Power rule: d/dx[x^n] = n*x^(n-1).  Constants factor out: d/dx[a*f(x)] = a*f'(x).  The derivative of sin(kx) is k*cos(kx) by the chain rule.",
    ),
    "integration": TopicReference(
        title="Integration",
        definition="Integration is the inverse of differentiation.  The indefinite integral ∫ f(x) dx is a family of antiderivatives differing by a constant C.",
        rules="Power rule for integration: ∫ x^n dx = x^(n+1)/(n+1) + C for n ≠ -1.  A definite integral ∫ from a to b f(x) dx equals F(b) - F(a) for any antiderivative F.",
    ),
    "geometry": TopicReference(
        title="Geometry",
        definition="Geometry studies shapes, their properties and their measurement.  Common measurements are area (the space inside a shape) and perimeter or circumference (its boundary length).",
        rules="Rectangle area = width * height; perimeter = 2*(width + height).  Triangle area = (1/2)*base*height.  Circle area = πr^2 and circumference = 2πr.",
    ),
    "probability": TopicReference(
        title="Probability",
        definition="Probability measures the likelihood of an event as a number between 0 and 1, equal to favourable outcomes divided by equally likely outcomes.",
        rules="For equally likely outcomes, P(event) = favourable / total.  Independent events multiply: P(A and B) = P(A)*P(B).  The number of k-subsets of an n-set is the binomial coefficient C(n, k).",
    ),
    "number_theory": TopicReference(
        title="Number Theory",
        definition="Number theory is the study of the integers, especially divisibility, primes and modular arithmetic.",
        rules="gcd(a, b) is the largest integer dividing both a and b; lcm(a, b) = a*b / gcd(a, b).  An integer greater than 1 is prime when it has no positive divisors other than 1 and itself.  a mod m is the remainder of a divided by m.",
    ),
    "trigonometry": TopicReference(
        title="Trigonometry",
        definition="Trigonometry relates the angles and side ratios of right triangles through the functions sine, cosine and tangent.",
        rules="Standard exact values: sin(30°) = 1/2, cos(45°) = √2/2, tan(60°) = √3.  sin(x) is positive in quadrants I and II; cos(x) is positive in quadrants I and IV.  The Pythagorean identity states sin^2(θ) + cos^2(θ) = 1.",
    ),
}
