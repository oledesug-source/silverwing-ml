"""Neural network layers with forward and backward pass."""

import math
import random

from .activations import Activation


class Parameter:
    def __init__(self, values: list[float], shape: tuple[int, ...]):
        self.values = list(values)
        self.shape = shape
        self.grad = [0.0] * len(values)

    def zero_grad(self):
        self.grad = [0.0] * len(self.values)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape})"


class Layer:
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.training = True

    def forward(self, x: list[float]) -> list[float]:
        raise NotImplementedError

    def backward(self, grad: list[float]) -> list[float]:
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(f"Linear({in_features}->{out_features})")
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        scale = math.sqrt(2.0 / in_features)
        self.weight = Parameter(
            [random.gauss(0, scale) for _ in range(in_features * out_features)],
            (out_features, in_features),
        )
        self.bias = Parameter([0.0] * out_features, (out_features,)) if bias else None
        self._last_input: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        self._last_input = list(x)
        out = []
        for i in range(self.out_features):
            s = 0.0
            for j in range(self.in_features):
                s += self.weight.values[i * self.in_features + j] * x[j]
            if self.use_bias:
                s += self.bias.values[i]
            out.append(s)
        return out

    def backward(self, grad: list[float]) -> list[float]:
        x = self._last_input
        new_grad = [0.0] * self.in_features
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.weight.grad[i * self.in_features + j] += grad[i] * x[j]
                new_grad[j] += self.weight.values[i * self.in_features + j] * grad[i]
            if self.use_bias:
                self.bias.grad[i] += grad[i]
        return new_grad

    def parameters(self) -> list[Parameter]:
        params = [self.weight]
        if self.use_bias:
            params.append(self.bias)
        return params


class ActivationLayer(Layer):
    def __init__(self, activation: Activation):
        super().__init__(f"Activation({activation.name})")
        self.activation = activation
        self._last_input: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        self._last_input = list(x)
        return [self.activation.fn(v) for v in x]

    def backward(self, grad: list[float]) -> list[float]:
        return [g * self.activation.derivative(v) for g, v in zip(grad, self._last_input)]


class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        super().__init__(f"Dropout(p={p})")
        self.p = p
        self._mask: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        if not self.training:
            return list(x)
        self._mask = [
            0.0 if random.random() < self.p else 1.0 / (1.0 - self.p)
            for _ in x
        ]
        return [v * m for v, m in zip(x, self._mask)]

    def backward(self, grad: list[float]) -> list[float]:
        if not self.training:
            return list(grad)
        return [g * m for g, m in zip(grad, self._mask)]


class BatchNorm1d(Layer):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__(f"BatchNorm1d({num_features})")
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter([1.0] * num_features, (num_features,))
        self.beta = Parameter([0.0] * num_features, (num_features,))
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self._last_input: list[float] = []
        self._last_normalized: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        self._last_input = list(x)
        if self.training:
            mean = sum(x) / len(x) if len(x) == self.num_features else x[0]
            var_vals = [(v - mean) ** 2 for v in x]
            var = sum(var_vals) / len(var_vals) if var_vals else 0.0
            self.running_mean = [
                (1 - self.momentum) * rm + self.momentum * mean
                for rm in self.running_mean
            ]
            self.running_var = [
                (1 - self.momentum) * rv + self.momentum * var
                for rv in self.running_var
            ]
        else:
            var_vals = self.running_var
        normalized = []
        for i, v in enumerate(x):
            m = self.running_mean[i] if not self.training else mean
            vr = self.running_var[i] if not self.training else var
            normalized.append((v - m) / math.sqrt(vr + self.eps))
        self._last_normalized = normalized
        return [
            self.gamma.values[i] * normalized[i] + self.beta.values[i]
            for i in range(self.num_features)
        ]

    def backward(self, grad: list[float]) -> list[float]:
        n = self.num_features
        new_grad = [0.0] * n
        for i in range(n):
            self.gamma.grad[i] += grad[i] * self._last_normalized[i]
            self.beta.grad[i] += grad[i]
            new_grad[i] = grad[i] * self.gamma.values[i]
        return new_grad

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]


class Sequential(Layer):
    def __init__(self, layers: list[Layer] | None = None):
        super().__init__("Sequential")
        self._layers = list(layers) if layers else []

    def add(self, layer: Layer) -> "Sequential":
        self._layers.append(layer)
        return self

    def forward(self, x: list[float]) -> list[float]:
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: list[float]) -> list[float]:
        g = grad
        for layer in reversed(self._layers):
            g = layer.backward(g)
        return g

    def parameters(self) -> list[Parameter]:
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        self.training = True
        for layer in self._layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self._layers:
            layer.eval()

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, idx: int) -> Layer:
        return self._layers[idx]

    def __iter__(self):
        return iter(self._layers)
