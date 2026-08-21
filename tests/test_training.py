"""Comprehensive tests for the training module."""

import math
import random

import pytest

from intelligence.training.activations import (
    ACTIVATIONS,
    RELU,
    SIGMOID,
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
from intelligence.training.data import (
    DataLoader,
    Dataset,
    k_fold_split,
    label_encode,
    normalize,
    one_hot,
    train_test_split,
)
from intelligence.training.losses import (
    BCE,
    MAE,
    MSE,
    binary_cross_entropy,
    cosine_similarity_loss,
    cross_entropy,
    get_loss,
    huber,
    kl_divergence,
    mae,
    mse,
    mse_derivative,
)
from intelligence.training.nn import (
    ActivationLayer,
    BatchNorm1d,
    Dropout,
    Linear,
    Parameter,
    Sequential,
)
from intelligence.training.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    RMSprop,
    get_optimizer,
)
from intelligence.training.training import (
    CosineAnnealingLR,
    EarlyStopping,
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

# ── Activation Tests ──────────────────────────────────────────────────────


class TestActivations:
    def test_sigmoid_zero(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-6

    def test_sigmoid_large_positive(self):
        assert abs(sigmoid(100.0) - 1.0) < 1e-6

    def test_sigmoid_large_negative(self):
        assert abs(sigmoid(-100.0)) < 1e-6

    def test_sigmoid_symmetry(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-6

    def test_sigmoid_derivative_at_zero(self):
        assert abs(sigmoid_derivative(0.0) - 0.25) < 1e-6

    def test_sigmoid_derivative_non_negative(self):
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert sigmoid_derivative(x) >= 0

    def test_tanh_zero(self):
        assert abs(tanh_fn(0.0)) < 1e-6

    def test_tanh_large(self):
        assert abs(tanh_fn(100.0) - 1.0) < 1e-6

    def test_tanh_derivative_at_zero(self):
        assert abs(tanh_derivative(0.0) - 1.0) < 1e-6

    def test_relu_zero(self):
        assert relu(0.0) == 0.0

    def test_relu_positive(self):
        assert relu(5.0) == 5.0

    def test_relu_negative(self):
        assert relu(-5.0) == 0.0

    def test_relu_derivative_positive(self):
        assert relu_derivative(5.0) == 1.0

    def test_relu_derivative_zero(self):
        assert relu_derivative(0.0) == 0.0

    def test_relu_derivative_negative(self):
        assert relu_derivative(-5.0) == 0.0

    def test_leaky_relu_positive(self):
        assert leaky_relu(5.0) == 5.0

    def test_leaky_relu_negative(self):
        assert abs(leaky_relu(-5.0) - (-0.05)) < 1e-6

    def test_elu_positive(self):
        assert elu(5.0) == 5.0

    def test_elu_negative(self):
        assert abs(elu(-1.0) - (math.exp(-1.0) - 1.0)) < 1e-6

    def test_softplus_positive(self):
        assert softplus(5.0) > 5.0

    def test_softplus_zero(self):
        assert abs(softplus(0.0) - math.log(2.0)) < 1e-6

    def test_swish_zero(self):
        assert abs(swish(0.0)) < 1e-6

    def test_swish_large(self):
        assert abs(swish(100.0) - 100.0) < 1e-3

    def test_identity(self):
        assert identity(5.0) == 5.0

    def test_softmax(self):
        result = softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-6
        assert result[2] > result[1] > result[0]

    def test_softmax_equal(self):
        result = softmax([1.0, 1.0, 1.0])
        for v in result:
            assert abs(v - 1 / 3) < 1e-6

    def test_get_activation(self):
        act = get_activation("relu")
        assert act.name == "relu"
        assert act(5.0) == 5.0

    def test_get_activation_unknown(self):
        with pytest.raises(ValueError):
            get_activation("unknown")

    def test_all_activations_callable(self):
        for _name, act in ACTIVATIONS.items():
            result = act(0.5)
            assert isinstance(result, float)

    def test_activation_repr(self):
        assert "sigmoid" in repr(SIGMOID)


# ── Loss Tests ────────────────────────────────────────────────────────────


class TestLosses:
    def test_mse_perfect(self):
        assert abs(mse([1.0, 2.0], [1.0, 2.0])) < 1e-6

    def test_mse_basic(self):
        assert abs(mse([1.0, 2.0], [2.0, 3.0]) - 1.0) < 1e-6

    def test_mse_derivative_perfect(self):
        grads = mse_derivative([1.0, 2.0], [1.0, 2.0])
        for g in grads:
            assert abs(g) < 1e-6

    def test_mse_derivative(self):
        grads = mse_derivative([3.0, 5.0], [1.0, 1.0])
        assert abs(grads[0] - 2.0) < 1e-6
        assert abs(grads[1] - 4.0) < 1e-6

    def test_mae_perfect(self):
        assert abs(mae([1.0, 2.0], [1.0, 2.0])) < 1e-6

    def test_mae_basic(self):
        assert abs(mae([1.0, 3.0], [2.0, 5.0]) - 1.5) < 1e-6

    def test_bce_perfect(self):
        assert abs(binary_cross_entropy([1.0], [1.0])) < 1e-6

    def test_bce_zero(self):
        assert abs(binary_cross_entropy([0.0], [0.0])) < 1e-6

    def test_bce_bad_prediction(self):
        assert binary_cross_entropy([0.001], [1.0]) > 5.0

    def test_cross_entropy_perfect(self):
        assert abs(cross_entropy([1.0], [1.0])) < 1e-6

    def test_huber_small_error(self):
        assert abs(huber([1.0], [1.5]) - 0.125) < 1e-6

    def test_huber_large_error(self):
        assert abs(huber([1.0], [5.0]) - 3.5) < 1e-6

    def test_kl_same(self):
        assert abs(kl_divergence([0.5, 0.5], [0.5, 0.5])) < 1e-6

    def test_kl_different(self):
        assert kl_divergence([0.25, 0.75], [0.5, 0.5]) > 0

    def test_cosine_identical(self):
        assert abs(cosine_similarity_loss([1.0, 0.0], [1.0, 0.0])) < 1e-6

    def test_cosine_opposite(self):
        assert abs(cosine_similarity_loss([1.0, 0.0], [-1.0, 0.0]) - 2.0) < 1e-6

    def test_get_loss(self):
        loss_fn = get_loss("mse")
        assert loss_fn.name == "mse"

    def test_get_loss_unknown(self):
        with pytest.raises(ValueError):
            get_loss("unknown")

    def test_loss_repr(self):
        assert "mse" in repr(MSE)

    def test_losses_all_callable(self):
        for _name, loss_fn in [("mse", MSE), ("mae", MAE), ("bce", BCE)]:
            result = loss_fn([0.5, 0.5], [1.0, 0.0])
            assert isinstance(result, float)


# ── Optimizer Tests ───────────────────────────────────────────────────────


class TestOptimizers:
    def test_sgd_basic(self):
        opt = SGD(learning_rate=0.1)
        params = [1.0, 2.0]
        grads = [1.0, 1.0]
        new_params = opt.step(params, grads)
        assert abs(new_params[0] - 0.9) < 1e-6
        assert abs(new_params[1] - 1.9) < 1e-6

    def test_sgd_momentum(self):
        opt = SGD(learning_rate=0.1, momentum=0.9)
        params = [1.0]
        grads = [1.0]
        new_params = opt.step(params, grads)
        assert abs(new_params[0] - 0.9) < 1e-6
        new_params2 = opt.step(new_params, grads)
        assert new_params2[0] < new_params[0]

    def test_sgd_weight_decay(self):
        opt = SGD(learning_rate=0.1, weight_decay=0.01)
        params = [1.0]
        grads = [0.0]
        new_params = opt.step(params, grads)
        assert new_params[0] < 1.0

    def test_adam_basic(self):
        opt = Adam(learning_rate=0.01)
        params = [1.0, 2.0]
        grads = [1.0, 1.0]
        new_params = opt.step(params, grads)
        assert new_params[0] < 1.0
        assert new_params[1] < 2.0

    def test_adam_convergence(self):
        opt = Adam(learning_rate=0.1)
        params = [5.0]
        for _ in range(100):
            grads = [2.0 * (params[0] - 0.0)]
            params = opt.step(params, grads)
        assert abs(params[0]) < 0.5

    def test_rmsprop_basic(self):
        opt = RMSprop(learning_rate=0.01)
        params = [1.0]
        grads = [1.0]
        new_params = opt.step(params, grads)
        assert new_params[0] < 1.0

    def test_adagrad_basic(self):
        opt = Adagrad(learning_rate=0.1)
        params = [1.0]
        grads = [1.0]
        new_params = opt.step(params, grads)
        assert new_params[0] < 1.0

    def test_adadelta_basic(self):
        opt = Adadelta(learning_rate=1.0)
        params = [1.0]
        grads = [1.0]
        new_params = opt.step(params, grads)
        assert isinstance(new_params[0], float)

    def test_reset(self):
        opt = SGD(learning_rate=0.1)
        opt.step([1.0], [1.0])
        opt.reset()
        assert opt.state == {}

    def test_get_optimizer(self):
        opt = get_optimizer("adam", learning_rate=0.001)
        assert isinstance(opt, Adam)
        assert opt.learning_rate == 0.001

    def test_get_optimizer_unknown(self):
        with pytest.raises(ValueError):
            get_optimizer("unknown")

    def test_optimizer_repr(self):
        opt = SGD(learning_rate=0.01)
        assert "SGD" in repr(opt)

    def test_all_optimizers_step(self):
        for name in ["sgd", "adam", "rmsprop", "adagrad", "adadelta"]:
            opt = get_optimizer(name, learning_rate=0.01)
            new_params = opt.step([1.0, 2.0], [0.1, 0.1])
            assert len(new_params) == 2


# ── Neural Network Tests ──────────────────────────────────────────────────


class TestNeuralNetwork:
    def test_parameter(self):
        p = Parameter([1.0, 2.0], (2,))
        assert p.values == [1.0, 2.0]
        assert p.shape == (2,)
        p.zero_grad()
        assert all(g == 0.0 for g in p.grad)

    def test_linear_forward(self):
        layer = Linear(3, 2)
        x = [1.0, 2.0, 3.0]
        out = layer.forward(x)
        assert len(out) == 2

    def test_linear_no_bias(self):
        layer = Linear(3, 2, bias=False)
        assert layer.bias is None
        assert len(layer.parameters()) == 1

    def test_linear_backward(self):
        layer = Linear(3, 2)
        x = [1.0, 0.0, 0.0]
        layer.forward(x)
        grad = layer.backward([1.0, 0.5])
        assert len(grad) == 3

    def test_linear_parameter_count(self):
        layer = Linear(4, 3)
        params = layer.parameters()
        assert len(params) == 2
        assert params[0].shape == (3, 4)
        assert params[1].shape == (3,)

    def test_activation_layer(self):
        layer = ActivationLayer(RELU)
        assert layer.forward([1.0, -1.0, 0.0]) == [1.0, 0.0, 0.0]

    def test_activation_layer_backward(self):
        layer = ActivationLayer(SIGMOID)
        layer.forward([0.0])
        grad = layer.backward([1.0])
        assert abs(grad[0] - 0.25) < 1e-6

    def test_dropout_train(self):
        layer = Dropout(p=0.0)
        layer.train()
        out = layer.forward([1.0, 2.0, 3.0])
        assert len(out) == 3

    def test_dropout_eval(self):
        layer = Dropout(p=0.5)
        layer.eval()
        out = layer.forward([1.0, 2.0, 3.0])
        assert out == [1.0, 2.0, 3.0]

    def test_batchnorm_forward(self):
        layer = BatchNorm1d(3)
        layer.train()
        x = [1.0, 2.0, 3.0]
        out = layer.forward(x)
        assert len(out) == 3

    def test_batchnorm_eval(self):
        layer = BatchNorm1d(3)
        layer.eval()
        x = [1.0, 2.0, 3.0]
        out = layer.forward(x)
        assert len(out) == 3

    def test_batchnorm_parameters(self):
        layer = BatchNorm1d(3)
        params = layer.parameters()
        assert len(params) == 2

    def test_sequential_forward(self):
        model = Sequential([
            Linear(3, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        out = model.forward([1.0, 2.0, 3.0])
        assert len(out) == 1

    def test_sequential_add(self):
        model = Sequential()
        model.add(Linear(3, 4))
        model.add(ActivationLayer(RELU))
        assert len(model) == 2

    def test_sequential_backward(self):
        model = Sequential([
            Linear(3, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        model.forward([1.0, 2.0, 3.0])
        grad = model.backward([1.0])
        assert len(grad) == 3

    def test_sequential_parameters(self):
        model = Sequential([
            Linear(3, 4),
            ActivationLayer(RELU),
            Linear(4, 2),
        ])
        params = model.parameters()
        assert len(params) == 4

    def test_sequential_getitem(self):
        layers = [Linear(3, 4), ActivationLayer(RELU)]
        model = Sequential(layers)
        assert model[0] is layers[0]
        assert model[1] is layers[1]

    def test_sequential_iter(self):
        layers = [Linear(3, 4), ActivationLayer(RELU)]
        model = Sequential(layers)
        assert list(model) == layers

    def test_sequential_train_eval(self):
        model = Sequential([Dropout(0.5)])
        model.train()
        assert model.training
        model.eval()
        assert not model.training
        assert not model[0].training


# ── Data Tests ────────────────────────────────────────────────────────────


class TestData:
    def test_dataset(self):
        ds = Dataset([[1, 2], [3, 4]], [0, 1])
        assert len(ds) == 2
        assert ds.n_features == 2
        assert ds[0] == ([1, 2], 0)

    def test_dataset_no_labels(self):
        ds = Dataset([[1, 2], [3, 4]])
        assert ds.y is None

    def test_dataset_subset(self):
        ds = Dataset([[1, 2], [3, 4], [5, 6]], [0, 1, 2])
        sub = ds.subset([0, 2])
        assert len(sub) == 2
        assert sub[0][0] == [1, 2]

    def test_dataloader(self):
        ds = Dataset([[1, 2], [3, 4], [5, 6]], [0, 1, 0])
        dl = DataLoader(ds, batch_size=2)
        batches = list(dl)
        assert len(batches) == 2
        assert len(batches[0][0]) == 2

    def test_dataloader_shuffle(self):
        ds = Dataset([[i] for i in range(10)], list(range(10)))
        dl = DataLoader(ds, batch_size=5, shuffle=True)
        batches = list(dl)
        assert len(batches) == 2

    def test_dataloader_drop_last(self):
        ds = Dataset([[i] for i in range(10)], list(range(10)))
        dl = DataLoader(ds, batch_size=3, drop_last=True)
        batches = list(dl)
        assert all(len(b[0]) == 3 for b in batches)

    def test_dataloader_len(self):
        ds = Dataset([[i] for i in range(10)], list(range(10)))
        dl = DataLoader(ds, batch_size=3)
        assert len(dl) == 4

    def test_train_test_split(self):
        X = [[i] for i in range(100)]
        y = list(range(100))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_train_test_split_no_labels(self):
        X = [[i] for i in range(100)]
        X_train, X_test = train_test_split(X, test_size=0.3, seed=42)
        assert len(X_train) == 70
        assert len(X_test) == 30

    def test_k_fold_split(self):
        folds = k_fold_split(10, k=5, seed=42)
        assert len(folds) == 5
        for train_idx, test_idx in folds:
            assert len(train_idx) + len(test_idx) == 10

    def test_k_fold_no_overlap(self):
        folds = k_fold_split(10, k=5, seed=42)
        all_test = []
        for _, test_idx in folds:
            all_test.extend(test_idx)
        assert len(all_test) == 10
        assert len(set(all_test)) == 10

    def test_normalize(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        X_norm, mean, std = normalize(X)
        assert len(X_norm) == 3
        assert len(X_norm[0]) == 2
        col0_mean = sum(X_norm[i][0] for i in range(3)) / 3
        col1_mean = sum(X_norm[i][1] for i in range(3)) / 3
        assert abs(col0_mean) < 1e-6
        assert abs(col1_mean) < 1e-6

    def test_normalize_with_params(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        mean = [2.0, 3.0]
        std = [1.0, 1.0]
        X_norm, _, _ = normalize(X, mean, std)
        assert abs(X_norm[0][0] + 1.0) < 1e-6

    def test_one_hot(self):
        labels = [0, 1, 2, 0]
        result = one_hot(labels, 3)
        assert result == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]

    def test_one_hot_auto_classes(self):
        labels = [0, 2, 1]
        result = one_hot(labels)
        assert len(result) == 3
        assert len(result[0]) == 3

    def test_label_encode(self):
        labels = ["cat", "dog", "cat", "bird"]
        encoded, mapping = label_encode(labels)
        assert encoded == [1, 2, 1, 0]
        assert mapping["bird"] == 0
        assert mapping["cat"] == 1
        assert mapping["dog"] == 2


# ── Training Tests ────────────────────────────────────────────────────────


class TestTraining:
    def test_history(self):
        h = TrainingHistory()
        h.record_train(1.0, 0.8, 0.01)
        h.record_val(1.2, 0.7)
        assert h.epochs == 1
        assert h.best_train_loss == 1.0
        assert h.best_val_loss == 1.2

    def test_early_stopping(self):
        es = EarlyStopping(patience=3)
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.91)
        assert not es.step(0.92)
        assert es.step(0.93)

    def test_early_stopping_reset(self):
        es = EarlyStopping(patience=1)
        es.step(1.0)
        es.step(1.1)
        assert es.should_stop
        es.reset()
        assert not es.should_stop

    def test_step_lr(self):
        opt = SGD(learning_rate=1.0)
        scheduler = StepLR(opt, step_size=2, gamma=0.5)
        lr = scheduler.step(0)
        assert lr == 1.0
        lr = scheduler.step(2)
        assert lr == 0.5

    def test_cosine_lr(self):
        opt = SGD(learning_rate=1.0)
        scheduler = CosineAnnealingLR(opt, T_max=10)
        lr_start = scheduler.step(0)
        lr_mid = scheduler.step(5)
        lr_end = scheduler.step(10)
        assert lr_start > lr_mid
        assert lr_end < lr_mid

    def test_reduce_on_plateau(self):
        opt = SGD(learning_rate=1.0)
        scheduler = ReduceOnPlateau(opt, factor=0.5, patience=2)
        lr = scheduler.step(1.0)
        assert lr == 1.0
        scheduler.step(1.1)
        scheduler.step(1.2)
        lr = scheduler.step(1.3)
        assert lr == 0.5

    def test_accuracy(self):
        assert accuracy([1, 0, 1], [1, 0, 0]) == 2 / 3
        assert accuracy([1, 1], [1, 1]) == 1.0
        assert accuracy([], []) == 0.0

    def test_mse_metric(self):
        preds = [[1.0, 2.0]]
        targets = [[1.0, 3.0]]
        assert abs(mse_metric(preds, targets) - 0.5) < 1e-6

    def test_cross_entropy_metric(self):
        preds = [[0.9, 0.1]]
        targets = [0]
        ce = cross_entropy_metric(preds, targets)
        assert ce < 0.2

    def test_train_one_epoch(self):
        model = Sequential([
            Linear(2, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [[3.0], [7.0], [11.0]]
        ds = Dataset(X, y)
        dl = DataLoader(ds, batch_size=2)
        loss = train_one_epoch(model, dl, MSE, SGD(learning_rate=0.001))
        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate(self):
        model = Sequential([
            Linear(2, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        X = [[1.0, 2.0], [3.0, 4.0]]
        y = [[3.0], [7.0]]
        ds = Dataset(X, y)
        dl = DataLoader(ds, batch_size=2)
        loss, metric = evaluate(model, dl, MSE)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_fit(self):
        random.seed(42)
        model = Sequential([
            Linear(1, 8),
            ActivationLayer(RELU),
            Linear(8, 1),
        ])
        X = [[i / 10.0] for i in range(20)]
        y = [[i / 10.0 * 2] for i in range(20)]
        ds = Dataset(X, y)
        history = fit(model, ds, epochs=20, batch_size=4, verbose=False)
        assert history.epochs == 20
        assert history.train_loss[-1] < history.train_loss[0]

    def test_fit_with_validation(self):
        random.seed(42)
        model = Sequential([
            Linear(1, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        X_train = [[i / 10.0] for i in range(20)]
        y_train = [[i / 10.0 * 2] for i in range(20)]
        X_val = [[i / 5.0] for i in range(5)]
        y_val = [[i / 5.0 * 2] for i in range(5)]
        ds_train = Dataset(X_train, y_train)
        ds_val = Dataset(X_val, y_val)
        history = fit(
            model, ds_train, val_dataset=ds_val,
            epochs=10, batch_size=4, verbose=False,
        )
        assert len(history.val_loss) == 10

    def test_fit_with_early_stopping(self):
        random.seed(42)
        model = Sequential([
            Linear(1, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        X = [[i / 10.0] for i in range(20)]
        y = [[i / 10.0 * 2] for i in range(20)]
        ds = Dataset(X, y)
        es = EarlyStopping(patience=3)
        history = fit(
            model, ds, val_dataset=ds,
            epochs=100, batch_size=4,
            early_stopping=es, verbose=False,
        )
        assert history.epochs < 100

    def test_fit_with_lr_scheduler(self):
        random.seed(42)
        model = Sequential([
            Linear(1, 4),
            ActivationLayer(RELU),
            Linear(4, 1),
        ])
        X = [[i / 10.0] for i in range(20)]
        y = [[i / 10.0 * 2] for i in range(20)]
        ds = Dataset(X, y)
        opt = Adam(learning_rate=0.01)
        scheduler = StepLR(opt, step_size=5, gamma=0.5)
        history = fit(
            model, ds, optimizer=opt,
            lr_scheduler=scheduler,
            epochs=10, batch_size=4, verbose=False,
        )
        assert history.epochs == 10
        assert history.learning_rates[-1] < 0.01
