"""Training loop and evaluation utilities."""

import math
from collections.abc import Callable

from .data import DataLoader, Dataset
from .losses import MSE, Loss
from .nn import Sequential
from .optimizers import SGD, Optimizer


class TrainingHistory:
    def __init__(self):
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_metric: list[float] = []
        self.val_metric: list[float] = []
        self.learning_rates: list[float] = []

    def record_train(self, loss: float, metric: float = 0.0, lr: float = 0.0):
        self.train_loss.append(loss)
        self.train_metric.append(metric)
        self.learning_rates.append(lr)

    def record_val(self, loss: float, metric: float = 0.0):
        self.val_loss.append(loss)
        self.val_metric.append(metric)

    @property
    def best_train_loss(self) -> float:
        return min(self.train_loss) if self.train_loss else float("inf")

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("inf")

    @property
    def epochs(self) -> int:
        return len(self.train_loss)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False


class LRScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr if initial_lr is not None else optimizer.learning_rate

    def step(self, epoch: int) -> float:
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int = 10, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch: int) -> float:
        lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        self.optimizer.learning_rate = lr
        return lr


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int = 50, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch: int) -> float:
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * epoch / self.T_max)
        ) / 2.0
        self.optimizer.learning_rate = lr
        return lr


class ReduceOnPlateau:
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 0.1,
        patience: int = 5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> float:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = max(self.optimizer.learning_rate * self.factor, self.min_lr)
                self.optimizer.learning_rate = new_lr
                self.counter = 0
        return self.optimizer.learning_rate


def accuracy(y_pred: list[int], y_true: list[int]) -> float:
    if len(y_pred) == 0:
        return 0.0
    return sum(1 for p, t in zip(y_pred, y_true) if p == t) / len(y_pred)


def mse_metric(y_pred: list[list[float]], y_true: list[list[float]]) -> float:
    total = 0.0
    n = 0
    for p_row, t_row in zip(y_pred, y_true):
        for p, t in zip(p_row, t_row):
            total += (p - t) ** 2
            n += 1
    return total / n if n > 0 else 0.0


def cross_entropy_metric(y_pred: list[list[float]], y_true: list[int]) -> float:
    eps = 1e-15
    total = 0.0
    for probs, label in zip(y_pred, y_true):
        p = max(eps, min(1.0 - eps, probs[label]))
        total -= math.log(p)
    return total / len(y_pred)


def train_one_epoch(
    model: Sequential,
    dataloader: DataLoader,
    loss_fn: Loss,
    optimizer: Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in dataloader:
        if y_batch is None:
            continue
        batch_loss = 0.0
        for x, y_true in zip(X_batch, y_batch):
            if isinstance(y_true, (int, float)):
                y_true_list = [y_true]
            else:
                y_true_list = list(y_true)
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y_true_list)
            batch_loss += loss
            grad = loss_fn.derivative(y_pred, y_true_list)
            model.backward(grad)
            params = model.parameters()
            grads = [sum(p.grad) / max(len(p.grad), 1) for p in params]
            params_values = [sum(p.values) / max(len(p.values), 1) for p in params]
            new_vals = optimizer.step(params_values, grads)
            for p, nv in zip(params, new_vals):
                for i in range(len(p.values)):
                    p.values[i] = nv
            for p in params:
                p.zero_grad()
        total_loss += batch_loss / len(X_batch)
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(
    model: Sequential,
    dataloader: DataLoader,
    loss_fn: Loss,
    metric_fn: Callable | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list = []
    all_targets: list = []
    for X_batch, y_batch in dataloader:
        if y_batch is None:
            continue
        batch_loss = 0.0
        for x, y_true in zip(X_batch, y_batch):
            if isinstance(y_true, (int, float)):
                y_true_list = [y_true]
            else:
                y_true_list = list(y_true)
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y_true_list)
            batch_loss += loss
            all_preds.append(y_pred)
            all_targets.append(y_true)
        total_loss += batch_loss / len(X_batch)
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    metric = 0.0
    if metric_fn and all_preds:
        metric = metric_fn(all_preds, all_targets)
    return avg_loss, metric


def fit(
    model: Sequential,
    train_dataset: Dataset,
    loss_fn: Loss = MSE,
    optimizer: Optimizer = None,
    epochs: int = 100,
    batch_size: int = 32,
    val_dataset: Dataset | None = None,
    lr_scheduler: LRScheduler | None = None,
    early_stopping: EarlyStopping | None = None,
    verbose: bool = False,
    metric_fn: Callable | None = None,
) -> TrainingHistory:
    if optimizer is None:
        optimizer = SGD(learning_rate=0.01)
    history = TrainingHistory()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        lr = optimizer.learning_rate
        train_metric = 0.0
        history.record_train(train_loss, train_metric, lr)
        if val_loader is not None:
            val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric_fn)
            history.record_val(val_loss, val_metric)
            if early_stopping and early_stopping.step(val_loss):
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        if lr_scheduler:
            lr_scheduler.step(epoch)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}"
            if val_loader:
                msg += f" - val_loss: {history.val_loss[-1]:.4f}"
            print(msg)
    return history
