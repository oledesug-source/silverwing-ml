"""Loss functions for neural network training."""

import math
from collections.abc import Callable


class Loss:
    def __init__(
        self,
        fn: Callable[[list[float], list[float]], float],
        derivative: Callable[[list[float], list[float]], list[float]],
        name: str = "",
    ):
        self.fn = fn
        self.derivative = derivative
        self.name = name or fn.__name__

    def __call__(self, predicted: list[float], actual: list[float]) -> float:
        return self.fn(predicted, actual)

    def __repr__(self) -> str:
        return f"Loss({self.name})"


def mse(predicted: list[float], actual: list[float]) -> float:
    n = len(predicted)
    return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / n


def mse_derivative(predicted: list[float], actual: list[float]) -> list[float]:
    n = len(predicted)
    return [2.0 * (p - a) / n for p, a in zip(predicted, actual)]


def mae(predicted: list[float], actual: list[float]) -> float:
    n = len(predicted)
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / n


def mae_derivative(predicted: list[float], actual: list[float]) -> list[float]:
    n = len(predicted)
    return [
        (1.0 if p > a else -1.0) / n if p != a else 0.0
        for p, a in zip(predicted, actual)
    ]


def binary_cross_entropy(predicted: list[float], actual: list[float]) -> float:
    eps = 1e-15
    n = len(predicted)
    total = 0.0
    for p, a in zip(predicted, actual):
        p = max(eps, min(1.0 - eps, p))
        total -= a * math.log(p) + (1.0 - a) * math.log(1.0 - p)
    return total / n


def binary_cross_entropy_derivative(
    predicted: list[float], actual: list[float]
) -> list[float]:
    eps = 1e-15
    n = len(predicted)
    return [
        (-a / max(eps, min(1.0 - eps, p)) + (1.0 - a) / (1.0 - max(eps, min(1.0 - eps, p)))) / n
        for p, a in zip(predicted, actual)
    ]


def cross_entropy(predicted: list[float], actual: list[float]) -> float:
    eps = 1e-15
    n = len(actual)
    total = 0.0
    for p, a in zip(predicted, actual):
        p = max(eps, min(1.0 - eps, p))
        total -= a * math.log(p)
    return total / n


def cross_entropy_derivative(
    predicted: list[float], actual: list[float]
) -> list[float]:
    eps = 1e-15
    n = len(actual)
    return [
        -a / max(eps, min(1.0 - eps, p)) / n
        for p, a in zip(predicted, actual)
    ]


def huber(predicted: list[float], actual: list[float], delta: float = 1.0) -> float:
    n = len(predicted)
    total = 0.0
    for p, a in zip(predicted, actual):
        err = abs(p - a)
        if err <= delta:
            total += 0.5 * err ** 2
        else:
            total += delta * err - 0.5 * delta ** 2
    return total / n


def huber_derivative(
    predicted: list[float], actual: list[float], delta: float = 1.0
) -> list[float]:
    n = len(predicted)
    grads = []
    for p, a in zip(predicted, actual):
        err = p - a
        if abs(err) <= delta:
            grads.append(err / n)
        else:
            grads.append(delta * (1.0 if err > 0 else -1.0) / n)
    return grads


def kl_divergence(predicted: list[float], actual: list[float]) -> float:
    eps = 1e-15
    n = len(predicted)
    total = 0.0
    for p, a in zip(predicted, actual):
        p = max(eps, p)
        a = max(eps, a)
        total += a * math.log(a / p)
    return total / n


def kl_divergence_derivative(
    predicted: list[float], actual: list[float]
) -> list[float]:
    eps = 1e-15
    n = len(predicted)
    return [
        -a / max(eps, p) / n
        for p, a in zip(predicted, actual)
    ]


def cosine_similarity_loss(predicted: list[float], actual: list[float]) -> float:
    dot = sum(p * a for p, a in zip(predicted, actual))
    norm_p = math.sqrt(sum(p ** 2 for p in predicted))
    norm_a = math.sqrt(sum(a ** 2 for a in actual))
    if norm_p == 0 or norm_a == 0:
        return 1.0
    return 1.0 - dot / (norm_p * norm_a)


def cosine_similarity_loss_derivative(
    predicted: list[float], actual: list[float]
) -> list[float]:
    dot = sum(p * a for p, a in zip(predicted, actual))
    norm_p = math.sqrt(sum(p ** 2 for p in predicted))
    norm_a = math.sqrt(sum(a ** 2 for a in actual))
    if norm_p == 0 or norm_a == 0:
        return [0.0] * len(predicted)
    n = len(predicted)
    return [
        (a / (norm_p * norm_a) - dot * p / (norm_p ** 3 * norm_a)) / n
        for p, a in zip(predicted, actual)
    ]


MSE = Loss(mse, mse_derivative, "mse")
MAE = Loss(mae, mae_derivative, "mae")
BCE = Loss(binary_cross_entropy, binary_cross_entropy_derivative, "bce")
CE = Loss(cross_entropy, cross_entropy_derivative, "cross_entropy")
HUBER = Loss(huber, huber_derivative, "huber")
KL = Loss(kl_divergence, kl_divergence_derivative, "kl_divergence")
COSINE = Loss(cosine_similarity_loss, cosine_similarity_loss_derivative, "cosine")

LOSSES = {
    "mse": MSE,
    "mae": MAE,
    "bce": BCE,
    "cross_entropy": CE,
    "huber": HUBER,
    "kl_divergence": KL,
    "cosine": COSINE,
}


def get_loss(name: str) -> Loss:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES)}")
    return LOSSES[name]
