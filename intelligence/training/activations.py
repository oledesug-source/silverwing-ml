"""Activation functions for neural networks."""

import math
from collections.abc import Callable


class Activation:
    def __init__(
        self,
        fn: Callable[[float], float],
        derivative: Callable[[float], float],
        name: str = "",
    ):
        self.fn = fn
        self.derivative = derivative
        self.name = name or fn.__name__

    def __call__(self, x: float) -> float:
        return self.fn(x)

    def __repr__(self) -> str:
        return f"Activation({self.name})"


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def sigmoid_derivative(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    return x if x > 0 else alpha * x


def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
    return 1.0 if x > 0 else alpha


def elu(x: float, alpha: float = 1.0) -> float:
    return x if x > 0 else alpha * (math.exp(x) - 1.0)


def elu_derivative(x: float, alpha: float = 1.0) -> float:
    return 1.0 if x > 0 else elu(x, alpha) + alpha


def softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log(1.0 + math.exp(x))


def softplus_derivative(x: float) -> float:
    return sigmoid(x)


def identity(x: float) -> float:
    return x


def identity_derivative(x: float) -> float:
    return 1.0


def swish(x: float) -> float:
    return x * sigmoid(x)


def swish_derivative(x: float) -> float:
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


SIGMOID = Activation(sigmoid, sigmoid_derivative, "sigmoid")
TANH = Activation(tanh_fn, tanh_derivative, "tanh")
RELU = Activation(relu, relu_derivative, "relu")
ELU = Activation(elu, elu_derivative, "elu")
SOFTPLUS = Activation(softplus, softplus_derivative, "softplus")
IDENTITY = Activation(identity, identity_derivative, "identity")
SWISH = Activation(swish, swish_derivative, "swish")

ACTIVATIONS = {
    "sigmoid": SIGMOID,
    "tanh": TANH,
    "relu": RELU,
    "elu": ELU,
    "softplus": SOFTPLUS,
    "identity": IDENTITY,
    "swish": SWISH,
}


def get_activation(name: str) -> Activation:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]


def softmax(values: list[float]) -> list[float]:
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]
