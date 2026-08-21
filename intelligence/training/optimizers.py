"""Optimizers for gradient descent."""

import math


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.state: dict[str, object] = {}

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        raise NotImplementedError

    def reset(self):
        self.state.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.learning_rate})"


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state["velocities"] = [0.0] * 1000

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        velocities = self.state["velocities"]
        if len(velocities) != len(params):
            self.state["velocities"] = [0.0] * len(params)
            velocities = self.state["velocities"]
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g + self.weight_decay * p
            velocities[i] = self.momentum * velocities[i] + g
            new_params.append(p - self.learning_rate * velocities[i])
        return new_params


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.state["m"] = []
        self.state["v"] = []
        self.state["t"] = 0

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        m = self.state["m"]
        v = self.state["v"]
        if len(m) != len(params):
            self.state["m"] = [0.0] * len(params)
            self.state["v"] = [0.0] * len(params)
            m = self.state["m"]
            v = self.state["v"]
        self.state["t"] = self.state["t"] + 1
        t = self.state["t"]
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g + self.weight_decay * p
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g
            m_hat = m[i] / (1.0 - self.beta1 ** t)
            v_hat = v[i] / (1.0 - self.beta2 ** t)
            new_params.append(p - self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon))
        return new_params


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.state["cache"] = []

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        cache = self.state["cache"]
        if len(cache) != len(params):
            self.state["cache"] = [0.0] * len(params)
            cache = self.state["cache"]
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g + self.weight_decay * p
            cache[i] = self.alpha * cache[i] + (1.0 - self.alpha) * g * g
            new_params.append(p - self.learning_rate * g / (math.sqrt(cache[i]) + self.epsilon))
        return new_params


class Adagrad(Optimizer):
    def __init__(
        self, learning_rate: float = 0.01, epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.state["cache"] = []

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        cache = self.state["cache"]
        if len(cache) != len(params):
            self.state["cache"] = [0.0] * len(params)
            cache = self.state["cache"]
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache[i] += g * g
            new_params.append(p - self.learning_rate * g / (math.sqrt(cache[i]) + self.epsilon))
        return new_params


class Adadelta(Optimizer):
    def __init__(
        self,
        learning_rate: float = 1.0,
        rho: float = 0.95,
        epsilon: float = 1e-6,
    ):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.state["cache_g"] = []
        self.state["cache_dx"] = []

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        cache_g = self.state["cache_g"]
        cache_dx = self.state["cache_dx"]
        if len(cache_g) != len(params):
            self.state["cache_g"] = [0.0] * len(params)
            self.state["cache_dx"] = [0.0] * len(params)
            cache_g = self.state["cache_g"]
            cache_dx = self.state["cache_dx"]
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache_g[i] = self.rho * cache_g[i] + (1.0 - self.rho) * g * g
            dx = -(math.sqrt(cache_dx[i] + self.epsilon) / math.sqrt(cache_g[i] + self.epsilon)) * g
            cache_dx[i] = self.rho * cache_dx[i] + (1.0 - self.rho) * dx * dx
            new_params.append(p + self.learning_rate * dx)
        return new_params


OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
}


def get_optimizer(name: str, **kwargs) -> Optimizer:
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](**kwargs)
