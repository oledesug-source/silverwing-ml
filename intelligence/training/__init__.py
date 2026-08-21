"""Training — neural networks, backpropagation, optimizers, data loading.

Pure-Python implementations for understanding ML training fundamentals.

Modules:
    activations: Activation functions and their derivatives
    losses: Loss functions (MSE, cross-entropy, etc.)
    optimizers: Gradient descent optimizers (SGD, Adam, RMSprop, etc.)
    nn: Neural network layers (Linear, BatchNorm, Dropout, Sequential)
    data: Dataset, DataLoader, train/test splitting, normalization
    training: Training loop, learning rate schedulers, evaluation
"""

from .activations import (
    ELU,
    IDENTITY,
    RELU,
    SIGMOID,
    SOFTPLUS,
    SWISH,
    TANH,
    Activation,
    elu,
    get_activation,
    identity,
    leaky_relu,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    softplus,
    swish,
    tanh_derivative,
    tanh_fn,
)
from .data import (
    DataLoader,
    Dataset,
    k_fold_split,
    label_encode,
    normalize,
    one_hot,
    train_test_split,
)
from .losses import (
    BCE,
    CE,
    COSINE,
    HUBER,
    KL,
    MAE,
    MSE,
    Loss,
    binary_cross_entropy,
    cosine_similarity_loss,
    cross_entropy,
    get_loss,
    huber,
    kl_divergence,
    mae,
    mae_derivative,
    mse,
    mse_derivative,
)
from .nn import (
    ActivationLayer,
    BatchNorm1d,
    Dropout,
    Layer,
    Linear,
    Parameter,
    Sequential,
)
from .optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Optimizer,
    RMSprop,
    get_optimizer,
)
from .training import (
    CosineAnnealingLR,
    EarlyStopping,
    LRScheduler,
    ReduceOnPlateau,
    StepLR,
    TrainingHistory,
    accuracy,
    cross_entropy_metric,
    evaluate,
    fit,
    mse_metric,
    train_one_epoch,
)

__all__ = [
    "Activation", "sigmoid", "sigmoid_derivative", "tanh_fn", "tanh_derivative",
    "relu", "relu_derivative", "leaky_relu", "elu", "softplus", "swish",
    "identity", "softmax",
    "SIGMOID", "TANH", "RELU", "ELU", "SOFTPLUS", "IDENTITY", "SWISH",
    "get_activation",
    "Loss", "mse", "mse_derivative", "mae", "mae_derivative",
    "binary_cross_entropy", "cross_entropy", "huber", "kl_divergence",
    "cosine_similarity_loss",
    "MSE", "MAE", "BCE", "CE", "HUBER", "KL", "COSINE", "get_loss",
    "Optimizer", "SGD", "Adam", "RMSprop", "Adagrad", "Adadelta", "get_optimizer",
    "Parameter", "Layer", "Linear", "ActivationLayer", "Dropout",
    "BatchNorm1d", "Sequential",
    "Dataset", "DataLoader", "train_test_split", "k_fold_split",
    "normalize", "one_hot", "label_encode",
    "TrainingHistory", "EarlyStopping", "LRScheduler", "StepLR",
    "CosineAnnealingLR", "ReduceOnPlateau",
    "accuracy", "mse_metric", "cross_entropy_metric",
    "train_one_epoch", "evaluate", "fit",
]
