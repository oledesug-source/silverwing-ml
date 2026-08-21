"""Probability distributions: continuous and discrete, each as a dataclass with pdf/pmf, cdf, sample, mean, variance, and stddev."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "UniformDistribution",
    "BernoulliDistribution",
    "BinomialDistribution",
    "PoissonDistribution",
    "GeometricDistribution",
    "NormalDistribution",
    "ExponentialDistribution",
    "GammaDistribution",
    "BetaDistribution",
    "ChiSquaredDistribution",
    "StudentTDistribution",
    "LogNormalDistribution",
    "ParetoDistribution",
]


def _factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _erf(x: float) -> float:
    """Horner-form rational approximation to the error function (Abramowitz & Stegun 7.1.26, max |error| < 1.5e-7)."""
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x
    t = 1.0 / (1.0 + 0.3275911 * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3 - 1.453152027 * t4 + 1.061405429 * t5
    return sign * (1.0 - poly * math.exp(-x * x))


def _gamma(z: float) -> float:
    """Lanczos approximation for the Gamma function."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1.0 - z))
    z -= 1.0
    g = 7
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return math.sqrt(2.0 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x


def _beta_incomplete(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function via continued fraction (Lentz's method)."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0 or x == 1.0:
        return x
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    f = 1.0
    c = 1.0
    d = 0.0
    for m in range(200):
        m_f = float(m)
        even = m % 2 == 0
        num: float
        if even:
            k = m_f / 2.0
            num = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1.0))
        else:
            k = (m_f - 1.0) / 2.0
            num = k * (b - k) * x / ((a + 2 * k) * (a + 2 * k + 1.0))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(c * d - 1.0) < 1e-10:
            break
    return front * (f - 1.0)


@dataclass(frozen=True)
class UniformDistribution:
    """Continuous uniform distribution on [a, b]."""

    a: float
    b: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        return 1.0 / (self.b - self.a) if self.a <= x <= self.b else 0.0

    def cdf(self, x: float) -> float:
        """Return the CDF at *x*."""
        if x < self.a:
            return 0.0
        if x > self.b:
            return 1.0
        return (x - self.a) / (self.b - self.a)

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one random sample."""
        r = rng or random
        return self.a + (self.b - self.a) * (r.random() if hasattr(r, "random") else r.random())

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return (self.a + self.b) / 2.0

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return (self.b - self.a) ** 2 / 12.0

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class BernoulliDistribution:
    """Bernoulli distribution with parameter p."""

    p: float

    def pmf(self, k: int) -> float:
        """Return the PMF at k ∈ {0, 1}."""
        if k == 1:
            return self.p
        if k == 0:
            return 1.0 - self.p
        return 0.0

    def cdf(self, k: float) -> float:
        """Return the CDF at k."""
        if k < 0:
            return 0.0
        if k < 1:
            return 1.0 - self.p
        return 1.0

    def sample(self, rng: random.Random | None = None) -> int:
        """Draw one Bernoulli trial."""
        r = rng or random
        return 1 if (r.random() if hasattr(r, "random") else r.random()) < self.p else 0

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self.p

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return self.p * (1.0 - self.p)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class BinomialDistribution:
    """Binomial distribution with parameters n and p."""

    n: int
    p: float

    def pmf(self, k: int) -> float:
        """Return the PMF at k."""
        if k < 0 or k > self.n:
            return 0.0
        return _factorial(self.n) / (_factorial(k) * _factorial(self.n - k)) * (self.p ** k) * ((1.0 - self.p) ** (self.n - k))

    def cdf(self, k: int) -> float:
        """Return the CDF at k (sum of PMF from 0 to k)."""
        return sum(self.pmf(i) for i in range(max(0, k + 1)))

    def sample(self, rng: random.Random | None = None) -> int:
        """Draw one sample using n independent Bernoulli trials."""
        r = rng or random
        count = 0
        for _ in range(self.n):
            val = r.random() if hasattr(r, "random") else r.random()
            if val < self.p:
                count += 1
        return count

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self.n * self.p

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return self.n * self.p * (1.0 - self.p)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class PoissonDistribution:
    """Poisson distribution with rate parameter λ."""

    lam: float

    def pmf(self, k: int) -> float:
        """Return the PMF at k."""
        if k < 0:
            return 0.0
        return (self.lam ** k) * math.exp(-self.lam) / _factorial(k)

    def cdf(self, k: int) -> float:
        """Return the CDF at k."""
        return sum(self.pmf(i) for i in range(max(0, k + 1)))

    def sample(self, rng: random.Random | None = None) -> int:
        """Draw one Poisson sample using the inverse-transform / Knuth method."""
        r = rng or random
        L = math.exp(-self.lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            val = r.random() if hasattr(r, "random") else r.random()
            p *= val
            if p <= L:
                break
        return k - 1

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self.lam

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return self.lam

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.lam)


@dataclass(frozen=True)
class GeometricDistribution:
    """Geometric distribution: number of trials until the first success."""

    p: float

    def pmf(self, k: int) -> float:
        """Return the PMF at k (k >= 1)."""
        if k < 1:
            return 0.0
        return ((1.0 - self.p) ** (k - 1)) * self.p

    def cdf(self, k: int) -> float:
        """Return the CDF at k."""
        if k < 1:
            return 0.0
        return 1.0 - ((1.0 - self.p) ** k)

    def sample(self, rng: random.Random | None = None) -> int:
        """Draw one sample."""
        r = rng or random
        u = r.random() if hasattr(r, "random") else r.random()
        return int(math.ceil(math.log(1.0 - u) / math.log(1.0 - self.p)))

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return 1.0 / self.p

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return (1.0 - self.p) / (self.p ** 2)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class NormalDistribution:
    """Normal (Gaussian) distribution with mean μ and std dev σ."""

    mu: float
    sigma: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        z = (x - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2.0 * math.pi))

    def cdf(self, x: float) -> float:
        """Return the CDF at *x* using the error function."""
        return 0.5 * (1.0 + _erf((x - self.mu) / (self.sigma * math.sqrt(2.0))))

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample using the Box-Muller transform."""
        r = rng or random
        u1 = r.random() if hasattr(r, "random") else r.random()
        u2 = r.random() if hasattr(r, "random") else r.random()
        while u1 == 0.0:
            u1 = r.random() if hasattr(r, "random") else r.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return self.mu + self.sigma * z

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self.mu

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return self.sigma ** 2

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return self.sigma


@dataclass(frozen=True)
class ExponentialDistribution:
    """Exponential distribution with rate λ."""

    lam: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        if x < 0:
            return 0.0
        return self.lam * math.exp(-self.lam * x)

    def cdf(self, x: float) -> float:
        """Return the CDF at *x*."""
        if x < 0:
            return 0.0
        return 1.0 - math.exp(-self.lam * x)

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample via inverse transform sampling."""
        r = rng or random
        u = r.random() if hasattr(r, "random") else r.random()
        return -math.log(1.0 - u) / self.lam

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return 1.0 / self.lam

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return 1.0 / (self.lam ** 2)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return 1.0 / self.lam


@dataclass(frozen=True)
class GammaDistribution:
    """Gamma distribution with shape α and rate β (parameterisation: f(x) = β^α x^(α-1) e^{-βx} / Γ(α))."""

    alpha: float
    beta: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        if x <= 0:
            return 0.0
        return (self.beta ** self.alpha) * (x ** (self.alpha - 1.0)) * math.exp(-self.beta * x) / _gamma(self.alpha)

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample using the Marsaglia & Tsang method for α ≥ 1; exponential shifting for α < 1."""
        r = rng or random
        if self.alpha < 1.0:
            return self._sample_alpha_lt_1(r)
        return self._sample_alpha_ge_1(r)

    def _sample_alpha_ge_1(self, r: random.Random) -> float:
        a = self.alpha
        d = a - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            while True:
                x = NormalDistribution(0.0, 1.0).sample(r)
                v = 1.0 + c * x
                if v > 0:
                    break
            v = v * v * v
            u = r.random() if hasattr(r, "random") else r.random()
            if u < 1.0 - 0.0331 * (x * x) * (x * x):
                return d * v / self.beta
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v / self.beta

    def _sample_alpha_lt_1(self, r: random.Random) -> float:
        u = r.random() if hasattr(r, "random") else r.random()
        return (self.sample(r) * (u ** (1.0 / self.alpha))) if False else self._do_alpha_lt_1(r)

    def _do_alpha_lt_1(self, r: random.Random) -> float:
        u = r.random() if hasattr(r, "random") else r.random()
        e = ExponentialDistribution(1.0).sample(r)
        return ((u ** (1.0 / self.alpha)) * e) / self.beta


@dataclass(frozen=True)
class BetaDistribution:
    """Beta distribution on [0, 1] with parameters α and β."""

    alpha: float
    beta: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        if x <= 0.0 or x >= 1.0:
            if x == 0.0 and self.alpha >= 1.0:
                pass
            elif x == 1.0 and self.beta >= 1.0:
                pass
            else:
                return 0.0
        lbeta = math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(self.alpha + self.beta)
        return math.exp((self.alpha - 1.0) * math.log(max(x, 1e-300)) + (self.beta - 1.0) * math.log(max(1.0 - x, 1e-300)) - lbeta)

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample using the Joehnk / gamma-ratio method."""
        r = rng or random
        ga = GammaDistribution(self.alpha, 1.0).sample(r)
        gb = GammaDistribution(self.beta, 1.0).sample(r)
        return ga / (ga + gb)

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class ChiSquaredDistribution:
    """Chi-squared distribution with *df* degrees of freedom."""

    df: int

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        if x <= 0:
            return 0.0
        k = float(self.df)
        half_k = k / 2.0
        return (x ** (half_k - 1.0)) * math.exp(-x / 2.0) / (2.0 ** half_k * _gamma(half_k))

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample by summing df squared standard normals."""
        r = rng or random
        total = 0.0
        n = NormalDistribution(0.0, 1.0)
        for _ in range(self.df):
            z = n.sample(r)
            total += z * z
        return total

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return float(self.df)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return 2.0 * float(self.df)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class StudentTDistribution:
    """Student's t-distribution with *df* degrees of freedom."""

    df: int

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        nu = float(self.df)
        coeff = _gamma((nu + 1.0) / 2.0) / (math.sqrt(nu * math.pi) * _gamma(nu / 2.0))
        return coeff * ((1.0 + x * x / nu) ** (-(nu + 1.0) / 2.0))

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample using the ratio-of-uniforms / normal-over-chi-squared method."""
        r = rng or random
        z = NormalDistribution(0.0, 1.0).sample(r)
        chi = ChiSquaredDistribution(self.df).sample(r)
        return z / math.sqrt(chi / float(self.df))

    @property
    def mean(self) -> float:
        """Mean of the distribution (0 for df > 1)."""
        if self.df <= 1:
            raise ValueError("mean undefined for df <= 1")
        return 0.0

    @property
    def variance(self) -> float:
        """Variance of the distribution (df/(df-2) for df > 2)."""
        if self.df <= 2:
            raise ValueError("variance undefined for df <= 2")
        return float(self.df) / float(self.df - 2)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class LogNormalDistribution:
    """Log-normal distribution: if X ~ Normal(μ, σ²) then e^X ~ LogNormal."""

    mu: float
    sigma: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        if x <= 0:
            return 0.0
        return math.exp(-0.5 * ((math.log(x) - self.mu) / self.sigma) ** 2) / (x * self.sigma * math.sqrt(2.0 * math.pi))

    def cdf(self, x: float) -> float:
        """Return the CDF at *x*."""
        if x <= 0:
            return 0.0
        return 0.5 * (1.0 + _erf((math.log(x) - self.mu) / (self.sigma * math.sqrt(2.0))))

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample by exponentiating a normal sample."""
        r = rng or random
        z = NormalDistribution(self.mu, self.sigma).sample(r)
        return math.exp(z)

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return math.exp(self.mu + self.sigma ** 2 / 2.0)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return (math.exp(self.sigma ** 2) - 1.0) * math.exp(2.0 * self.mu + self.sigma ** 2)

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class ParetoDistribution:
    """Pareto distribution with shape α and minimum scale x_m."""

    alpha: float
    xm: float

    def pdf(self, x: float) -> float:
        """Return the PDF at *x*."""
        if x < self.xm:
            return 0.0
        return (self.alpha * (self.xm ** self.alpha)) / (x ** (self.alpha + 1.0))

    def cdf(self, x: float) -> float:
        """Return the CDF at *x*."""
        if x < self.xm:
            return 0.0
        return 1.0 - (self.xm / x) ** self.alpha

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample via inverse transform sampling."""
        r = rng or random
        u = r.random() if hasattr(r, "random") else r.random()
        return self.xm / ((1.0 - u) ** (1.0 / self.alpha))

    @property
    def mean(self) -> float:
        """Mean of the distribution (α > 1)."""
        if self.alpha <= 1:
            raise ValueError("mean undefined for alpha <= 1")
        return (self.alpha * self.xm) / (self.alpha - 1.0)

    @property
    def variance(self) -> float:
        """Variance of the distribution (α > 2)."""
        if self.alpha <= 2:
            raise ValueError("variance undefined for alpha <= 2")
        return (self.xm ** 2 * self.alpha) / ((self.alpha - 1.0) ** 2 * (self.alpha - 2.0))

    @property
    def stddev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance)
