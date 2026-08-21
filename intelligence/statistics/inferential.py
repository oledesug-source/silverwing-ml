"""Inferential statistics: hypothesis tests, confidence intervals, effect sizes, resampling methods."""

from __future__ import annotations

import math
import random

__all__ = [
    "z_score",
    "confidence_interval",
    "hypothesis_test_z",
    "hypothesis_test_t",
    "t_test_independent",
    "paired_t_test",
    "chi_square_test",
    "anova_one_way",
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "bootstrap_confidence",
    "permutation_test",
    "EffectSize",
]


def _mean(data: list[float]) -> float:
    return sum(data) / len(data)


def _std(data: list[float], ddof: int = 1) -> float:
    n = len(data)
    mu = _mean(data)
    ss = sum((x - mu) ** 2 for x in data)
    return math.sqrt(ss / (n - ddof))


def _variance(data: list[float], ddof: int = 1) -> float:
    n = len(data)
    mu = _mean(data)
    ss = sum((x - mu) ** 2 for x in data)
    return ss / (n - ddof)


def _erf(x: float) -> float:
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
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1.0 - z))
    z -= 1.0
    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return math.sqrt(2.0 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x


def _t_cdf(t_val: float, df: int) -> float:
    x = float(df) / (float(df) + t_val * t_val)
    a = float(df) / 2.0
    b = 0.5
    ib = _regularised_beta_incomplete(x, a, b)
    if t_val >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def _regularised_beta_incomplete(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    f = 1.0
    c = 1.0
    d = 0.0
    for m in range(200):
        m_f = float(m)
        even = m % 2 == 0
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


def _chi_square_cdf(x: float, k: int) -> float:
    if x <= 0:
        return 0.0
    a = float(k) / 2.0
    return _regularised_gamma_lower(a, x / 2.0)


def _regularised_gamma_lower(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        ap = a
        s = 1.0 / a
        ds = 1.0 / a
        for _ in range(200):
            ap += 1.0
            ds *= x / ap
            s += ds
            if abs(ds) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for _ in range(200):
        an = -_frac_part(a, _)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return 1.0 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _frac_part(a: float, n: int) -> float:
    if n == 0:
        return a
    return -(a + float(n)) * (a + float(n) - 1.0)


def z_score(data_point: float, mean: float, std: float) -> float:
    """Compute the z-score of a data point given the population mean and standard deviation."""
    if std == 0:
        raise ValueError("standard deviation must be non-zero")
    return (data_point - mean) / std


def confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute a confidence interval for the mean using the t-distribution."""
    n = len(data)
    if n < 2:
        raise ValueError("need at least two data points")
    mu = _mean(data)
    s = _std(data)
    se = s / math.sqrt(n)
    alpha = 1.0 - confidence
    df = n - 1
    t_crit = _inverse_t_cdf(1.0 - alpha / 2.0, df)
    margin = t_crit * se
    return (mu - margin, mu + margin)


def _inverse_t_cdf(p: float, df: int) -> float:
    """Approximate the inverse t CDF via bisection then Newton refinement."""
    if df == 1:
        return math.tan(math.pi * (p - 0.5))
    if df == 2:
        return (2.0 * p - 1.0) / math.sqrt(2.0 * p * (1.0 - p))
    lo, hi = -15.0, 15.0
    if p < 0.5:
        hi = 0.0
    elif p > 0.5:
        lo = 0.0
    else:
        return 0.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2.0
    for _ in range(50):
        err = _t_cdf(x, df) - p
        if abs(err) < 1e-10:
            break
        nu = float(df)
        pdf_val = _gamma((nu + 1.0) / 2.0) / (math.sqrt(nu * math.pi) * _gamma(nu / 2.0)) * ((1.0 + x * x / nu) ** (-(nu + 1.0) / 2.0))
        if pdf_val < 1e-30:
            break
        x -= err / pdf_val
    return x


def hypothesis_test_z(sample_mean: float, pop_mean: float, std: float, n: int) -> tuple[float, float]:
    """Perform a one-sample z-test and return (z_stat, p_value)."""
    if std == 0 or n == 0:
        raise ValueError("std and n must be non-zero")
    se = std / math.sqrt(n)
    z = (sample_mean - pop_mean) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / math.sqrt(2.0))))
    return (z, p)


def hypothesis_test_t(data: list[float], pop_mean: float) -> tuple[float, float]:
    """Perform a one-sample t-test and return (t_stat, p_value)."""
    n = len(data)
    if n < 2:
        raise ValueError("need at least two data points")
    mu = _mean(data)
    s = _std(data)
    se = s / math.sqrt(n)
    t = (mu - pop_mean) / se
    p = 2.0 * (1.0 - _t_cdf(abs(t), n - 1))
    return (t, p)


def t_test_independent(sample1: list[float], sample2: list[float]) -> tuple[float, float]:
    """Perform Welch's independent two-sample t-test and return (t_stat, p_value)."""
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        raise ValueError("need at least two data points per sample")
    m1, m2 = _mean(sample1), _mean(sample2)
    v1, v2 = _variance(sample1), _variance(sample2)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return (0.0, 1.0)
    t = (m1 - m2) / se
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = int(num / denom) if denom > 0 else 1
    df = max(df, 1)
    p = 2.0 * (1.0 - _t_cdf(abs(t), df))
    return (t, p)


def paired_t_test(sample1: list[float], sample2: list[float]) -> tuple[float, float]:
    """Perform a paired sample t-test and return (t_stat, p_value)."""
    if len(sample1) != len(sample2):
        raise ValueError("samples must have the same length")
    diffs = [a - b for a, b in zip(sample1, sample2)]
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least two pairs")
    mu_d = _mean(diffs)
    s_d = _std(diffs)
    se = s_d / math.sqrt(n)
    if se == 0:
        return (0.0, 1.0)
    t = mu_d / se
    p = 2.0 * (1.0 - _t_cdf(abs(t), n - 1))
    return (t, p)


def chi_square_test(observed: list[float], expected: list[float]) -> tuple[float, float]:
    """Perform a chi-square goodness-of-fit test and return (chi2_stat, p_value)."""
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have the same length")
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e == 0:
            raise ValueError("expected frequencies must be non-zero")
        chi2 += (o - e) ** 2 / e
    df = len(observed) - 1
    p = 1.0 - _chi_square_cdf(chi2, df) if df > 0 else 1.0
    return (chi2, p)


def anova_one_way(*groups: list[float]) -> tuple[float, float]:
    """Perform a one-way ANOVA and return (f_stat, p_value)."""
    if len(groups) < 2:
        raise ValueError("need at least two groups")
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = _mean([x for g in groups for x in g])
    ssb = sum(len(g) * (_mean(g) - grand_mean) ** 2 for g in groups)
    ssw = sum(sum((x - _mean(g)) ** 2 for x in g) for g in groups)
    dfb = k - 1
    dfw = N - k
    if dfw == 0 or dfw < 0:
        raise ValueError("insufficient data for ANOVA")
    msb = ssb / dfb
    msw = ssw / dfw
    if msw == 0:
        return (float("inf"), 0.0)
    f = msb / msw
    p = 1.0 - _f_cdf(f, dfb, dfw)
    return (f, p)


def _f_cdf(x: float, d1: int, d2: int) -> float:
    if x <= 0:
        return 0.0
    v1 = float(d1)
    v2 = float(d2)
    y = v1 * x / (v1 * x + v2)
    return _regularised_beta_incomplete(y, v1 / 2.0, v2 / 2.0)


def mann_whitney_u(sample1: list[float], sample2: list[float]) -> tuple[float, float]:
    """Perform the Mann-Whitney U test and return (u_stat, p_value)."""
    n1, n2 = len(sample1), len(sample2)
    combined = [(v, 1) for v in sample1] + [(v, 2) for v in sample2]
    combined_sorted = sorted(combined, key=lambda t: t[0])
    ranks = [0.0] * len(combined_sorted)
    i = 0
    while i < len(combined_sorted):
        j = i
        while j < len(combined_sorted) - 1 and combined_sorted[j + 1][0] == combined_sorted[j][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(ranks[i] for i in range(len(combined_sorted)) if combined_sorted[i][1] == 1)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu_u = n1 * n2 / 2.0
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_u == 0:
        return (u, 1.0)
    z = (u - mu_u) / sigma_u
    p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / math.sqrt(2.0))))
    return (u, p)


def wilcoxon_signed_rank(sample1: list[float], sample2: list[float]) -> tuple[float, float]:
    """Perform the Wilcoxon signed-rank test and return (W_stat, p_value)."""
    if len(sample1) != len(sample2):
        raise ValueError("samples must have the same length")
    diffs = [(a - b, i) for i, (a, b) in enumerate(zip(sample1, sample2)) if a != b]
    if not diffs:
        return (0.0, 1.0)
    diffs_abs = sorted(diffs, key=lambda t: abs(t[0]))
    ranks = [0.0] * len(diffs_abs)
    i = 0
    while i < len(diffs_abs):
        j = i
        while j < len(diffs_abs) - 1 and abs(diffs_abs[j + 1][0]) == abs(diffs_abs[j][0]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    w_pos = sum(ranks[i] for i in range(len(diffs_abs)) if diffs_abs[i][0] > 0)
    w_neg = sum(ranks[i] for i in range(len(diffs_abs)) if diffs_abs[i][0] < 0)
    w = min(w_pos, w_neg)
    n = len(diffs_abs)
    if n >= 20:
        mu_w = n * (n + 1) / 4.0
        sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if sigma_w == 0:
            return (w, 1.0)
        z = (w - mu_w) / sigma_w
        p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / math.sqrt(2.0))))
    else:
        p = 1.0
    return (w, p)


def bootstrap_confidence(data: list[float], stat_func, n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a statistic."""
    rng = random.Random()
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_func(sample))
    boot_stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(math.floor(alpha * n_bootstrap))
    hi_idx = int(math.floor((1.0 - alpha) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return (boot_stats[lo_idx], boot_stats[hi_idx])


def permutation_test(sample1: list[float], sample2: list[float], stat_func, n_permutations: int = 1000) -> float:
    """Perform a two-sample permutation test and return the p-value."""
    rng = random.Random()
    combined = list(sample1) + list(sample2)
    n1 = len(sample1)
    observed_stat = stat_func(sample1, sample2)
    count = 0
    for _ in range(n_permutations):
        perm = combined[:]
        rng.shuffle(perm)
        s1 = perm[:n1]
        s2 = perm[n1:]
        perm_stat = stat_func(s1, s2)
        if abs(perm_stat) >= abs(observed_stat):
            count += 1
    return count / n_permutations


class EffectSize:
    """Effect size measures: Cohen's d, Hedges' g, and Glass's delta."""

    @staticmethod
    def cohen_d(sample1: list[float], sample2: list[float]) -> float:
        """Compute Cohen's d between two independent samples."""
        n1, n2 = len(sample1), len(sample2)
        m1, m2 = _mean(sample1), _mean(sample2)
        v1, v2 = _variance(sample1), _variance(sample2)
        pooled_std = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (m1 - m2) / pooled_std

    @staticmethod
    def hedges_g(sample1: list[float], sample2: list[float]) -> float:
        """Compute Hedges' g (bias-corrected Cohen's d)."""
        d = EffectSize.cohen_d(sample1, sample2)
        n1, n2 = len(sample1), len(sample2)
        df = n1 + n2 - 2
        correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0 else 1.0
        return d * correction

    @staticmethod
    def glass_delta(sample1: list[float], sample2: list[float]) -> float:
        """Compute Glass's delta (uses control group std dev)."""
        m1, m2 = _mean(sample1), _mean(sample2)
        s2 = _std(sample2)
        if s2 == 0:
            return 0.0
        return (m1 - m2) / s2
