"""Comprehensive test suite for the intelligence.statistics module.

Covers distributions, descriptive statistics, inference, regression, probability theory, and time series.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.statistics.descriptive import (
    central_moment,
    correlation,
    covariance,
    frequency_distribution,
    geometric_mean,
    harmonic_mean,
    iqr,
    kurtosis,
    mean,
    median,
    mode,
    moment,
    percentile,
    quartiles,
    range_stats,
    skewness,
    spearman_correlation,
    std_dev,
    summary,
    trimmed_mean,
    variance,
    weighted_mean,
)
from intelligence.statistics.distributions import (
    BernoulliDistribution,
    BetaDistribution,
    BinomialDistribution,
    ChiSquaredDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GeometricDistribution,
    LogNormalDistribution,
    NormalDistribution,
    ParetoDistribution,
    PoissonDistribution,
    StudentTDistribution,
    UniformDistribution,
)
from intelligence.statistics.inferential import (
    EffectSize,
    anova_one_way,
    bootstrap_confidence,
    chi_square_test,
    confidence_interval,
    hypothesis_test_t,
    hypothesis_test_z,
    mann_whitney_u,
    paired_t_test,
    permutation_test,
    t_test_independent,
    wilcoxon_signed_rank,
    z_score,
)
from intelligence.statistics.probability import (
    Combinatorics,
    MarkovChain,
    MonteCarlo,
    ProbabilitySpace,
    RandomWalk,
)
from intelligence.statistics.regression import (
    lasso_regression,
    linear_regression,
    logistic_regression,
    multiple_regression,
    polynomial_regression,
    predict,
    residual_analysis,
    ridge_regression,
)
from intelligence.statistics.time_series import (
    auto_correlation,
    exponential_moving_average,
    exponential_smoothing,
    forecast_simple,
    holt_linear,
    moving_average,
    partial_auto_correlation,
    stationarity_test,
    trend_decomposition,
)

passed = 0
failed = 0
errors = []


def check(condition: bool, name: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        errors.append(name)
        print(f"  FAIL  {name}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ── Distribution Tests ──────────────────────────────────────────────

def test_uniform_distribution():
    print("\n--- Uniform Distribution ---")
    d = UniformDistribution(0.0, 10.0)
    check(approx(d.mean, 5.0), "uniform mean")
    check(approx(d.variance, 100.0 / 12.0), "uniform variance")
    check(approx(d.stddev, math.sqrt(100.0 / 12.0)), "uniform stddev")
    check(approx(d.pdf(5.0), 0.1), "uniform pdf(5)")
    check(approx(d.pdf(-1.0), 0.0), "uniform pdf(-1) = 0")
    check(approx(d.cdf(5.0), 0.5), "uniform cdf(5)")
    check(0.0 <= d.sample() <= 10.0, "uniform sample in range")


def test_bernoulli_distribution():
    print("\n--- Bernoulli Distribution ---")
    d = BernoulliDistribution(0.7)
    check(approx(d.mean, 0.7), "bernoulli mean")
    check(approx(d.variance, 0.21), "bernoulli variance")
    check(approx(d.pmf(1), 0.7), "bernoulli pmf(1)")
    check(approx(d.pmf(0), 0.3), "bernoulli pmf(0)")
    check(approx(d.cdf(0.5), 0.3), "bernoulli cdf(0.5)")
    s = d.sample()
    check(s in (0, 1), "bernoulli sample is 0 or 1")


def test_binomial_distribution():
    print("\n--- Binomial Distribution ---")
    d = BinomialDistribution(n=10, p=0.5)
    check(approx(d.mean, 5.0), "binomial mean")
    check(approx(d.variance, 2.5), "binomial variance")
    check(approx(d.pmf(5), math.comb(10, 5) * 0.5**10), "binomial pmf(5)")
    check(0 <= d.sample() <= 10, "binomial sample in range")


def test_poisson_distribution():
    print("\n--- Poisson Distribution ---")
    d = PoissonDistribution(lam=3.0)
    check(approx(d.mean, 3.0), "poisson mean")
    check(approx(d.variance, 3.0), "poisson variance")
    check(approx(d.pmf(0), math.exp(-3.0)), "poisson pmf(0)")
    check(approx(d.pmf(1), 3.0 * math.exp(-3.0)), "poisson pmf(1)")
    check(d.sample() >= 0, "poisson sample >= 0")


def test_geometric_distribution():
    print("\n--- Geometric Distribution ---")
    d = GeometricDistribution(p=0.5)
    check(approx(d.mean, 2.0), "geometric mean")
    check(approx(d.variance, 2.0), "geometric variance")
    check(approx(d.pmf(1), 0.5), "geometric pmf(1)")
    check(d.sample() >= 1, "geometric sample >= 1")


def test_normal_distribution():
    print("\n--- Normal Distribution ---")
    d = NormalDistribution(mu=0.0, sigma=1.0)
    check(approx(d.mean, 0.0), "normal mean")
    check(approx(d.variance, 1.0), "normal variance")
    check(approx(d.pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi)), "normal pdf(0)")
    check(approx(d.cdf(0.0), 0.5), "normal cdf(0)")
    check(approx(d.cdf(1.96), 0.975, tol=0.01), "normal cdf(1.96) ~ 0.975")
    rng = random.Random(42)
    sample = [d.sample(rng) for _ in range(10000)]
    s_mean = sum(sample) / len(sample)
    check(abs(s_mean) < 0.1, "normal sample mean near 0")


def test_exponential_distribution():
    print("\n--- Exponential Distribution ---")
    d = ExponentialDistribution(lam=2.0)
    check(approx(d.mean, 0.5), "exponential mean")
    check(approx(d.variance, 0.25), "exponential variance")
    check(approx(d.pdf(0), 2.0), "exponential pdf(0)")
    check(approx(d.cdf(0), 0.0), "exponential cdf(0) = 0")
    check(d.sample() >= 0, "exponential sample >= 0")


def test_gamma_distribution():
    print("\n--- Gamma Distribution ---")
    d = GammaDistribution(alpha=2.0, beta=1.0)
    check(approx(d.pdf(1.0), math.exp(-1.0), tol=0.01), "gamma pdf(1)")
    s = d.sample()
    check(s > 0, "gamma sample > 0")


def test_beta_distribution():
    print("\n--- Beta Distribution ---")
    d = BetaDistribution(alpha=2.0, beta=5.0)
    check(approx(d.mean, 2.0 / 7.0), "beta mean")
    check(approx(d.variance, (2.0 * 5.0) / (49.0 * 8.0)), "beta variance")
    s = d.sample()
    check(0.0 <= s <= 1.0, "beta sample in [0,1]")


def test_chi_squared_distribution():
    print("\n--- Chi-Squared Distribution ---")
    d = ChiSquaredDistribution(df=5)
    check(approx(d.mean, 5.0), "chi2 mean")
    check(approx(d.variance, 10.0), "chi2 variance")
    s = d.sample()
    check(s >= 0, "chi2 sample >= 0")


def test_student_t_distribution():
    print("\n--- Student-t Distribution ---")
    d = StudentTDistribution(df=10)
    check(approx(d.mean, 0.0), "t mean")
    check(approx(d.variance, 10.0 / 8.0), "t variance")
    check(approx(d.pdf(0.0), StudentTDistribution(df=10).pdf(0.0)), "t pdf symmetric")


def test_lognormal_distribution():
    print("\n--- Log-Normal Distribution ---")
    d = LogNormalDistribution(mu=0.0, sigma=0.5)
    check(d.mean > 1.0, "lognormal mean > 1")
    check(d.cdf(0.0) == 0.0, "lognormal cdf(0) = 0")
    s = d.sample()
    check(s > 0, "lognormal sample > 0")


def test_pareto_distribution():
    print("\n--- Pareto Distribution ---")
    d = ParetoDistribution(alpha=3.0, xm=1.0)
    check(approx(d.mean, 1.5), "pareto mean")
    check(d.cdf(1.0) == 0.0, "pareto cdf(xm) = 0")
    check(d.cdf(2.0) > 0, "pareto cdf(2) > 0")
    s = d.sample()
    check(s >= 1.0, "pareto sample >= xm")


# ── Descriptive Statistics Tests ────────────────────────────────────

def test_descriptive_mean():
    print("\n--- Descriptive: mean ---")
    check(approx(mean([1, 2, 3, 4, 5]), 3.0), "mean of 1..5")
    check(approx(mean([10.0]), 10.0), "mean of single element")


def test_descriptive_median():
    print("\n--- Descriptive: median ---")
    check(approx(median([1, 3, 5, 7, 9]), 5.0), "median odd")
    check(approx(median([1, 2, 3, 4]), 2.5), "median even")
    check(approx(median([42]), 42.0), "median single")


def test_descriptive_mode():
    print("\n--- Descriptive: mode ---")
    m = mode([1, 1, 2, 3, 3])
    check(1 in m and 3 in m, "mode bimodal")


def test_descriptive_weighted_mean():
    print("\n--- Descriptive: weighted_mean ---")
    check(approx(weighted_mean([1, 2, 3], [1, 1, 1]), 2.0), "weighted mean equal weights")
    check(approx(weighted_mean([1, 2, 3], [0, 0, 1]), 3.0), "weighted mean one weight")


def test_descriptive_geometric_harmonic():
    print("\n--- Descriptive: geometric & harmonic mean ---")
    check(approx(geometric_mean([1, 2, 4]), 2.0), "geometric mean")
    check(approx(harmonic_mean([1, 2, 4]), 12.0 / 7.0), "harmonic mean")


def test_descriptive_trimmed_mean():
    print("\n--- Descriptive: trimmed_mean ---")
    check(approx(trimmed_mean([1, 2, 3, 4, 100], 0.2), 3.0), "trimmed mean")


def test_descriptive_percentile_quartiles():
    print("\n--- Descriptive: percentile & quartiles ---")
    data = list(range(1, 101))
    check(approx(percentile(data, 50), 50.5), "percentile 50")
    q1, q2, q3 = quartiles(data)
    check(approx(q1, 25.75, tol=1.0), "Q1 ~ 26")
    check(approx(q2, 50.5), "Q2 = median")
    check(approx(q3, 75.25, tol=1.0), "Q3 ~ 75")


def test_descriptive_iqr_range():
    print("\n--- Descriptive: iqr & range ---")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    check(approx(iqr(data), 4.5, tol=0.5), "iqr")
    lo, hi = range_stats(data)
    check(lo == 1 and hi == 10, "range_stats")


def test_descriptive_variance_std():
    print("\n--- Descriptive: variance & std_dev ---")
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    check(approx(variance(data), 32.0 / 7.0, tol=0.001), "sample variance")
    check(approx(variance(data, population=True), 4.0), "pop variance")
    check(approx(std_dev(data), math.sqrt(32.0 / 7.0), tol=0.001), "sample std")
    check(approx(std_dev(data, population=True), 2.0), "pop std")


def test_descriptive_covariance_correlation():
    print("\n--- Descriptive: covariance & correlation ---")
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 5.0, 4.0, 5.0]
    check(approx(covariance(x, y), 1.5), "covariance")
    r = correlation(x, y)
    check(-1.0 <= r <= 1.0, "correlation in [-1,1]")


def test_descriptive_spearman():
    print("\n--- Descriptive: spearman_correlation ---")
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    r = spearman_correlation(x, y)
    check(approx(r, -1.0), "spearman perfect negative")


def test_descriptive_skewness_kurtosis():
    print("\n--- Descriptive: skewness & kurtosis ---")
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    check(approx(skewness(data), 0.0, tol=0.1), "skewness symmetric")
    check(approx(kurtosis(data), -1.2, tol=0.2), "kurtosis uniform-like")


def test_descriptive_moment():
    print("\n--- Descriptive: moment & central_moment ---")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    check(approx(moment(data, 1), 3.0), "1st raw moment")
    check(approx(central_moment(data, 2), 2.0), "2nd central moment = var (pop)")


def test_descriptive_frequency_summary():
    print("\n--- Descriptive: frequency_distribution & summary ---")
    data = [1, 1, 2, 3, 3, 3]
    freq = frequency_distribution(data)
    check(freq[3] == 3, "frequency mode count")
    s = summary(data)
    check(approx(s["mean"], mean(data)), "summary mean")
    check(s["count"] == 6, "summary count")


# ── Inferential Statistics Tests ────────────────────────────────────

def test_z_score():
    print("\n--- Inferential: z_score ---")
    check(approx(z_score(75, 50, 10), 2.5), "z-score")


def test_confidence_interval():
    print("\n--- Inferential: confidence_interval ---")
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lo, hi = confidence_interval(data, 0.95)
    check(lo < mean(data) < hi, "CI contains mean")


def test_hypothesis_test_z():
    print("\n--- Inferential: hypothesis_test_z ---")
    z, p = hypothesis_test_z(sample_mean=52.0, pop_mean=50.0, std=10.0, n=30)
    check(z > 0, "z-stat positive")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_hypothesis_test_t():
    print("\n--- Inferential: hypothesis_test_t ---")
    data = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 4.9, 5.0, 5.1]
    t, p = hypothesis_test_t(data, pop_mean=5.0)
    check(abs(t) < 3.0, "t-stat small for close mean")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_t_test_independent():
    print("\n--- Inferential: t_test_independent ---")
    s1 = [5.0, 6.0, 7.0, 8.0, 9.0]
    s2 = [3.0, 4.0, 5.0, 6.0, 7.0]
    t, p = t_test_independent(s1, s2)
    check(t > 0, "t-test independent t > 0")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_paired_t_test():
    print("\n--- Inferential: paired_t_test ---")
    pre = [80.0, 85.0, 90.0, 75.0, 88.0]
    post = [82.0, 87.0, 92.0, 78.0, 90.0]
    t, p = paired_t_test(pre, post)
    check(abs(t) < 10.0, "paired t-stat reasonable")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_chi_square_test():
    print("\n--- Inferential: chi_square_test ---")
    obs = [20.0, 30.0, 50.0]
    exp = [25.0, 25.0, 50.0]
    chi2, p = chi_square_test(obs, exp)
    check(chi2 > 0, "chi2 stat > 0")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_anova_one_way():
    print("\n--- Inferential: anova_one_way ---")
    g1 = [5.0, 6.0, 7.0, 8.0]
    g2 = [10.0, 11.0, 12.0, 13.0]
    g3 = [15.0, 16.0, 17.0, 18.0]
    f, p = anova_one_way(g1, g2, g3)
    check(f > 1.0, "ANOVA F > 1 for different groups")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_mann_whitney_u():
    print("\n--- Inferential: mann_whitney_u ---")
    s1 = [1, 2, 3, 4, 5]
    s2 = [6, 7, 8, 9, 10]
    u, p = mann_whitney_u(s1, s2)
    check(u >= 0, "U stat >= 0")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_wilcoxon_signed_rank():
    print("\n--- Inferential: wilcoxon_signed_rank ---")
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    s2 = [1.1, 2.1, 3.1, 4.1, 5.1]
    w, p = wilcoxon_signed_rank(s1, s2)
    check(w >= 0, "W stat >= 0")
    check(0.0 <= p <= 1.0, "p-value in [0,1]")


def test_bootstrap_confidence():
    print("\n--- Inferential: bootstrap_confidence ---")
    data = list(range(1, 101))
    lo, hi = bootstrap_confidence(data, mean, n_bootstrap=500, confidence=0.95)
    check(lo < hi, "CI lower < upper")
    check(lo < 50.5 < hi, "CI contains population mean ~50.5")


def test_permutation_test():
    print("\n--- Inferential: permutation_test ---")
    s1 = [1.0, 2.0, 3.0]
    s2 = [10.0, 11.0, 12.0]

    def diff_means(a, b):
        return mean(a) - mean(b)

    p = permutation_test(s1, s2, diff_means, n_permutations=200)
    check(0.0 <= p <= 1.0, "permutation p-value in [0,1]")


def test_effect_size():
    print("\n--- Inferential: EffectSize ---")
    s1 = [5.0, 6.0, 7.0, 8.0]
    s2 = [1.0, 2.0, 3.0, 4.0]
    d = EffectSize.cohen_d(s1, s2)
    check(d > 0, "Cohen's d > 0")
    g = EffectSize.hedges_g(s1, s2)
    check(g > 0, "Hedges' g > 0")
    delta = EffectSize.glass_delta(s1, s2)
    check(delta > 0, "Glass's delta > 0")


# ── Regression Tests ────────────────────────────────────────────────

def test_linear_regression():
    print("\n--- Regression: linear_regression ---")
    x = [float(i) for i in range(10)]
    y = [2.0 * xi + 3.0 for xi in x]
    res = linear_regression(x, y)
    check(approx(res.slope, 2.0), "linear slope = 2")
    check(approx(res.intercept, 3.0), "linear intercept = 3")
    check(approx(res.r_squared, 1.0), "linear R² = 1")


def test_multiple_regression():
    print("\n--- Regression: multiple_regression ---")
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    y = [5.0, 8.0, 11.0, 14.0, 17.0]
    res = multiple_regression(X, y)
    check(len(res.coefficients) == 2, "multiple reg has 2 coefs")
    check(res.r_squared > 0.9, "multiple reg R² > 0.9")


def test_polynomial_regression():
    print("\n--- Regression: polynomial_regression ---")
    x = [float(i) for i in range(-5, 6)]
    y = [xi ** 2 for xi in x]
    res = polynomial_regression(x, y, degree=2)
    check(res.r_squared > 0.99, "poly deg-2 R² ~ 1 for x²")


def test_logistic_regression():
    print("\n--- Regression: logistic_regression ---")
    X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    res = logistic_regression(X, y, lr=0.1, epochs=2000)
    check(res.accuracy >= 0.75, "logistic accuracy >= 0.75")
    check(len(res.coefficients) == 1, "logistic has 1 coef")
    check(len(res.loss_history) == 2000, "logistic loss history length")


def test_ridge_regression():
    print("\n--- Regression: ridge_regression ---")
    x = [float(i) for i in range(10)]
    y = [3.0 * xi + 1.0 for xi in x]
    res = ridge_regression(x, y, alpha=0.1)
    check(approx(res.slope, 3.0, tol=0.1), "ridge slope ~ 3")
    check(approx(res.intercept, 1.0, tol=0.1), "ridge intercept ~ 1")


def test_lasso_regression():
    print("\n--- Regression: lasso_regression ---")
    x = [float(i) for i in range(10)]
    y = [4.0 * xi + 2.0 for xi in x]
    res = lasso_regression(x, y, alpha=0.01, lr=0.001, epochs=5000)
    check(res.slope > 0, "lasso slope > 0")
    check(res.r_squared > 0.9, "lasso R² > 0.9")


def test_residual_analysis():
    print("\n--- Regression: residual_analysis ---")
    x = [float(i) for i in range(10)]
    y = [2.0 * xi + 1.0 for xi in x]
    res = linear_regression(x, y)
    diag = residual_analysis(res)
    check("durbin_watson" in diag, "residual has durbin_watson")
    check("skewness" in diag, "residual has skewness")


def test_predict():
    print("\n--- Regression: predict ---")
    x = [float(i) for i in range(10)]
    y = [2.0 * xi + 1.0 for xi in x]
    res = linear_regression(x, y)
    preds = predict(res, [[0.0], [5.0], [10.0]])
    check(approx(preds[0], 1.0), "predict(0) = 1")
    check(approx(preds[1], 11.0), "predict(5) = 11")


# ── Probability Theory Tests ────────────────────────────────────────

def test_probability_space():
    print("\n--- Probability: ProbabilitySpace ---")
    ps = ProbabilitySpace()
    ps.add_outcome("H", 0.5)
    ps.add_outcome("T", 0.5)
    ps.add_event("Heads", {"H"})
    ps.add_event("Tails", {"T"})
    check(approx(ps.probability("Heads"), 0.5), "P(Heads)")
    check(approx(ps.probability("Tails"), 0.5), "P(Tails)")


def test_combinatorics():
    print("\n--- Probability: Combinatorics ---")
    c = Combinatorics()
    check(c.factorial(5) == 120, "5! = 120")
    check(c.permutation(5, 2) == 20, "P(5,2) = 20")
    check(c.combination(5, 2) == 10, "C(5,2) = 10")
    check(c.derangement(4) == 9, "!4 = 9")
    check(c.catalan_number(3) == 5, "C_3 = 5")
    check(c.stirling_number_second(4, 2) == 7, "S(4,2) = 7")


def test_markov_chain():
    print("\n--- Probability: MarkovChain ---")
    mc = MarkovChain(
        states=["Sunny", "Rainy"],
        transition_matrix=[[0.8, 0.2], [0.4, 0.6]],
    )
    ss = mc.steady_state
    check(approx(ss["Sunny"], 2.0 / 3.0, tol=0.01), "steady state Sunny")
    check(approx(ss["Rainy"], 1.0 / 3.0, tol=0.01), "steady state Rainy")
    path = mc.simulate("Sunny", steps=5, rng=random.Random(42))
    check(len(path) == 6, "simulate path length")
    check(path[0] == "Sunny", "simulate starts at Sunny")


def test_random_walk():
    print("\n--- Probability: RandomWalk ---")
    path_1d = RandomWalk.simulate_1d(steps=100, p=0.5, rng=random.Random(42))
    check(len(path_1d) == 101, "1D path length")
    check(path_1d[0] == 0, "1D starts at 0")
    path_2d = RandomWalk.simulate_2d(steps=50, rng=random.Random(42))
    check(len(path_2d) == 51, "2D path length")


def test_monte_carlo():
    print("\n--- Probability: MonteCarlo ---")
    mc = MonteCarlo(rng=random.Random(42))
    pi_est = mc.estimate_pi(n_samples=100_000)
    check(abs(pi_est - math.pi) < 0.05, "MC pi estimate close")


# ── Time Series Tests ───────────────────────────────────────────────

def test_moving_average():
    print("\n--- TimeSeries: moving_average ---")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ma = moving_average(data, window=3)
    check(len(ma) == 5, "moving average length")
    check(approx(ma[-1], 4.0), "MA(3) last value")


def test_exponential_moving_average():
    print("\n--- TimeSeries: exponential_moving_average ---")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ema = exponential_moving_average(data, alpha=0.5)
    check(len(ema) == 5, "EMA length")
    check(approx(ema[0], 1.0), "EMA first value")


def test_exponential_smoothing():
    print("\n--- TimeSeries: exponential_smoothing ---")
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    es = exponential_smoothing(data, alpha=0.3)
    check(len(es) == 5, "ES length")
    check(es[-1] < data[-1] or approx(es[-1], data[-1]), "ES <= last value for upward trend")


def test_holt_linear():
    print("\n--- TimeSeries: holt_linear ---")
    data = [float(i * 2) for i in range(10)]
    hl = holt_linear(data, alpha=0.5, beta=0.5)
    check(len(hl) == 10, "Holt linear length")
    check(hl[-1] > data[-2], "Holt linear forecasts upward for upward trend")


def test_auto_correlation():
    print("\n--- TimeSeries: auto_correlation ---")
    data = [float(i) for i in range(100)]
    acf1 = auto_correlation(data, lag=1)
    check(acf1 > 0.9, "ACF(1) near 1 for linear series")
    acf0 = auto_correlation(data, lag=0)
    check(approx(acf0, 1.0), "ACF(0) = 1")


def test_partial_auto_correlation():
    print("\n--- TimeSeries: partial_auto_correlation ---")
    data = [float(i % 2) for i in range(100)]
    pacf1 = partial_auto_correlation(data, lag=1)
    check(-1.0 <= pacf1 <= 1.0, "PACF(1) in [-1,1]")


def test_stationarity_test():
    print("\n--- TimeSeries: stationarity_test ---")
    rng = random.Random(42)
    data = [rng.gauss(0, 1) for _ in range(200)]
    result = stationarity_test(data)
    check("test_statistic" in result, "ADF has test_statistic")
    check("stationary" in result, "ADF has stationary flag")


def test_trend_decomposition():
    print("\n--- TimeSeries: trend_decomposition ---")
    data = [10.0 + 2.0 * i + (1.0 if i % 7 == 0 else 0.0) for i in range(50)]
    decomp = trend_decomposition(data)
    check(len(decomp["trend"]) == 50, "trend length")
    check(len(decomp["seasonal"]) == 50, "seasonal length")
    check(len(decomp["residual"]) == 50, "residual length")
    total = [decomp["trend"][i] + decomp["seasonal"][i] + decomp["residual"][i] for i in range(50)]
    for i in range(50):
        check(approx(total[i], data[i], tol=1e-10), f"decomposition sum matches at {i}")


def test_forecast_simple():
    print("\n--- TimeSeries: forecast_simple ---")
    data = [10.0, 11.0, 12.0, 13.0, 14.0]
    fc = forecast_simple(data, steps=3)
    check(len(fc) == 3, "forecast length")
    check(all(f > 0 for f in fc), "forecast positive")


# ── Run All Tests ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Silverwing-ML intelligence.statistics — Test Suite")
    print("=" * 60)

    test_uniform_distribution()
    test_bernoulli_distribution()
    test_binomial_distribution()
    test_poisson_distribution()
    test_geometric_distribution()
    test_normal_distribution()
    test_exponential_distribution()
    test_gamma_distribution()
    test_beta_distribution()
    test_chi_squared_distribution()
    test_student_t_distribution()
    test_lognormal_distribution()
    test_pareto_distribution()

    test_descriptive_mean()
    test_descriptive_median()
    test_descriptive_mode()
    test_descriptive_weighted_mean()
    test_descriptive_geometric_harmonic()
    test_descriptive_trimmed_mean()
    test_descriptive_percentile_quartiles()
    test_descriptive_iqr_range()
    test_descriptive_variance_std()
    test_descriptive_covariance_correlation()
    test_descriptive_spearman()
    test_descriptive_skewness_kurtosis()
    test_descriptive_moment()
    test_descriptive_frequency_summary()

    test_z_score()
    test_confidence_interval()
    test_hypothesis_test_z()
    test_hypothesis_test_t()
    test_t_test_independent()
    test_paired_t_test()
    test_chi_square_test()
    test_anova_one_way()
    test_mann_whitney_u()
    test_wilcoxon_signed_rank()
    test_bootstrap_confidence()
    test_permutation_test()
    test_effect_size()

    test_linear_regression()
    test_multiple_regression()
    test_polynomial_regression()
    test_logistic_regression()
    test_ridge_regression()
    test_lasso_regression()
    test_residual_analysis()
    test_predict()

    test_probability_space()
    test_combinatorics()
    test_markov_chain()
    test_random_walk()
    test_monte_carlo()

    test_moving_average()
    test_exponential_moving_average()
    test_exponential_smoothing()
    test_holt_linear()
    test_auto_correlation()
    test_partial_auto_correlation()
    test_stationarity_test()
    test_trend_decomposition()
    test_forecast_simple()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if errors:
        print(f"  FAILED:  {', '.join(errors)}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
