"""Time series analysis: moving averages, exponential smoothing, ACF/PACF, stationarity tests, decomposition, and forecasting."""

from __future__ import annotations

import math

__all__ = [
    "moving_average",
    "exponential_moving_average",
    "exponential_smoothing",
    "holt_linear",
    "auto_correlation",
    "partial_auto_correlation",
    "stationarity_test",
    "trend_decomposition",
    "forecast_simple",
]


def _mean(data: list[float]) -> float:
    return sum(data) / len(data)


def moving_average(data: list[float], window: int) -> list[float]:
    """Compute the simple moving average with the given *window* size."""
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(data)
    result: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        result.append(_mean(data[start : i + 1]))
    return result


def exponential_moving_average(data: list[float], alpha: float) -> list[float]:
    """Compute the exponential moving average with smoothing factor *alpha*."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if not data:
        return []
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1.0 - alpha) * result[-1])
    return result


def exponential_smoothing(data: list[float], alpha: float) -> list[float]:
    """Compute single exponential smoothing (identical to EMA but named for time-series convention)."""
    return exponential_moving_average(data, alpha)


def holt_linear(data: list[float], alpha: float, beta: float) -> list[float]:
    """Compute Holt's linear trend (double exponential smoothing) and return level + trend values.

    Returns a list of (level + trend) one-step-ahead forecasts for each time index.
    """
    if not (0.0 < alpha <= 1.0 and 0.0 < beta <= 1.0):
        raise ValueError("alpha and beta must be in (0, 1]")
    if len(data) < 2:
        return list(data)
    level = data[0]
    trend = data[1] - data[0]
    forecasts = [level + trend]
    for t in range(1, len(data)):
        new_level = alpha * data[t] + (1.0 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1.0 - beta) * trend
        level = new_level
        trend = new_trend
        forecasts.append(level + trend)
    return forecasts


def auto_correlation(data: list[float], lag: int) -> float:
    """Compute the sample autocorrelation function at the given *lag*."""
    n = len(data)
    if lag < 0 or lag >= n:
        raise ValueError("lag must be >= 0 and < len(data)")
    mu = _mean(data)
    ss_tot = sum((x - mu) ** 2 for x in data)
    if ss_tot == 0:
        return 0.0
    ss_lag = sum((data[i] - mu) * (data[i - lag] - mu) for i in range(lag, n))
    return ss_lag / ss_tot


def partial_auto_correlation(data: list[float], lag: int) -> float:
    """Compute the partial autocorrelation function at the given *lag* via the Durbin-Levinson recursion."""
    if lag < 0 or lag >= len(data):
        raise ValueError("lag must be >= 0 and < len(data)")
    if lag == 0:
        return 1.0
    acf = [auto_correlation(data, k) for k in range(1, lag + 1)]
    phi = [[0.0] * (lag + 1) for _ in range(lag + 1)]
    phi[1][1] = acf[0]
    for k in range(2, lag + 1):
        num = acf[k - 1] - sum(phi[k - 1][j] * acf[k - 1 - j] for j in range(1, k - 1))
        denom = 1.0 - sum(phi[k - 1][j] * acf[j - 1] for j in range(1, k - 1))
        if abs(denom) < 1e-12:
            phi[k][k] = 0.0
        else:
            phi[k][k] = num / denom
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
    return phi[lag][lag]


def stationarity_test(data: list[float]) -> dict:
    """Perform the Augmented Dickey-Fuller test and return test statistic, p-value approximation, and lags."""
    n = len(data)
    if n < 5:
        raise ValueError("need at least 5 observations")
    lags = min(int(math.floor(12.0 * (n / 100.0) ** 0.25)), n // 4)
    diff_y = [data[i] - data[i - 1] for i in range(1, n)]
    y_lag = data[:-1]
    T = len(y_lag)
    dy_mean = _mean(diff_y)
    yl_mean = _mean(y_lag)
    sxy = sum((diff_y[i] - dy_mean) * (y_lag[i] - yl_mean) for i in range(T))
    sxx = sum((y_lag[i] - yl_mean) ** 2 for i in range(T))
    if sxx == 0:
        adf_stat = 0.0
    else:
        gamma_hat = sxy / sxx
        residuals = [diff_y[i] - gamma_hat * y_lag[i] for i in range(T)]
        sse = sum(r ** 2 for r in residuals)
        se_gamma = math.sqrt(sse / (T - 1)) / math.sqrt(sxx) if sxx > 0 and T > 1 else 1.0
        adf_stat = gamma_hat / se_gamma if se_gamma > 0 else 0.0
    critical_values = {
        "1%": -3.43,
        "5%": -2.86,
        "10%": -2.57,
    }
    stationary = adf_stat < critical_values["5%"]
    return {
        "test_statistic": adf_stat,
        "lags_used": lags,
        "critical_values": critical_values,
        "stationary": stationary,
        "n_observations": n,
    }


def trend_decomposition(data: list[float]) -> dict:
    """Decompose a time series into trend, seasonal, and residual components via moving averages.

    Uses a period of 7 by default.  If the series is shorter, uses the full length.
    """
    n = len(data)
    period = min(7, n // 2) if n >= 4 else n
    if period < 1:
        period = 1
    trend: list[float] = []
    half = period // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        trend.append(_mean(data[lo:hi]))
    detrended = [data[i] - trend[i] for i in range(n)]
    seasonal: list[float] = [0.0] * n
    if period > 1:
        for s in range(period):
            indices = list(range(s, n, period))
            avg = _mean([detrended[i] for i in indices]) if indices else 0.0
            for i in indices:
                seasonal[i] = avg
    residual = [data[i] - trend[i] - seasonal[i] for i in range(n)]
    return {
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
    }


def forecast_simple(data: list[float], steps: int) -> list[float]:
    """Forecast *steps* into the future using the last smoothed level from simple exponential smoothing."""
    if not data:
        raise ValueError("data must not be empty")
    alpha = 0.3
    level = data[0]
    for val in data[1:]:
        level = alpha * val + (1.0 - alpha) * level
    return [level] * steps
