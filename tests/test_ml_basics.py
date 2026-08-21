"""Comprehensive tests for the ml_basics module."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import random

from intelligence.ml_basics.clustering import (
    DBSCAN,
    GMM,
    AgglomerativeClustering,
    HierarchicalClustering,
    KMeans,
    KMeansPlusPlus,
    SpectralClustering,
)
from intelligence.ml_basics.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    VotingClassifier,
)
from intelligence.ml_basics.feature_selection import (
    CorrelationFilter,
    SelectKBest,
    VarianceThreshold,
    chi_squared,
    mutual_information,
)
from intelligence.ml_basics.linear_models import (
    ElasticNet,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    MultiClassLogisticRegression,
    RidgeRegression,
)
from intelligence.ml_basics.metrics import (
    accuracy,
    adjusted_r2,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mae,
    map_score,
    mse,
    ndcg,
    precision,
    r2_score,
    recall,
    rmse,
    roc_auc_score,
    silhouette_score,
)
from intelligence.ml_basics.neighbors import (
    BallTree,
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
    RadiusNeighborsClassifier,
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
)
from intelligence.ml_basics.preprocessing import (
    KFold,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from intelligence.ml_basics.tree_models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    entropy,
    gini_impurity,
)


def _make_regression(n=100, features=3, noise=0.1):
    rng = random.Random(42)
    X = [[rng.gauss(0, 1) for _ in range(features)] for _ in range(n)]
    y = [sum(x[i] * (i + 1) for i in range(features)) + rng.gauss(0, noise) for x in X]
    return X, y


def _make_classification(n=100, features=2):
    rng = random.Random(42)
    X = [[rng.gauss(0, 1) for _ in range(features)] for _ in range(n)]
    y = [1 if sum(x) > 0 else 0 for x in X]
    return X, y


def _make_multiclass(n=120, features=2):
    rng = random.Random(42)
    X = [[rng.gauss(0, 1) for _ in range(features)] for _ in range(n)]
    y = []
    for x in X:
        s = sum(x)
        if s > 1:
            y.append(2)
        elif s > -1:
            y.append(1)
        else:
            y.append(0)
    return X, y


class TestLinearModels(unittest.TestCase):

    def test_linear_regression_fit_predict(self):
        X, y = _make_regression(50, 3, 0.01)
        lr = LinearRegression()
        lr.fit(X, y)
        preds = lr.predict(X)
        self.assertEqual(len(preds), 50)
        for p in preds:
            self.assertIsInstance(p, float)

    def test_linear_regression_score(self):
        X, y = _make_regression(50, 3, 0.01)
        lr = LinearRegression()
        lr.fit(X, y)
        s = lr.score(X, y)
        self.assertGreater(s, 0.9)

    def test_linear_regression_coefficients(self):
        X, y = _make_regression(50, 2, 0.001)
        lr = LinearRegression()
        lr.fit(X, y)
        self.assertEqual(len(lr.coefficients), 2)
        self.assertAlmostEqual(lr.coefficients[0], 1.0, delta=0.3)
        self.assertAlmostEqual(lr.coefficients[1], 2.0, delta=0.3)

    def test_ridge_regression(self):
        X, y = _make_regression(50, 3, 0.1)
        ridge = RidgeRegression(alpha=0.1)
        ridge.fit(X, y)
        s = ridge.score(X, y)
        self.assertGreater(s, 0.5)

    def test_lasso_regression(self):
        X, y = _make_regression(50, 3, 0.1)
        lasso = LassoRegression(alpha=0.01, lr=0.01, epochs=500)
        lasso.fit(X, y)
        preds = lasso.predict(X)
        self.assertEqual(len(preds), 50)

    def test_elastic_net(self):
        X, y = _make_regression(50, 3, 0.1)
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, lr=0.01, epochs=500)
        en.fit(X, y)
        preds = en.predict(X)
        self.assertEqual(len(preds), 50)

    def test_logistic_regression_fit_predict(self):
        X, y = _make_classification(50, 2)
        lr = LogisticRegression(lr=0.5, epochs=200)
        lr.fit(X, y)
        preds = lr.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreater(acc, 0.7)

    def test_logistic_regression_proba(self):
        X, y = _make_classification(50, 2)
        lr = LogisticRegression(lr=0.5, epochs=200)
        lr.fit(X, y)
        probas = lr.predict_proba(X)
        self.assertEqual(len(probas), 50)
        for p in probas:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_multi_class_logistic(self):
        X, y = _make_multiclass(60, 2)
        mcl = MultiClassLogisticRegression(lr=0.5, epochs=200)
        mcl.fit(X, y)
        preds = mcl.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreater(acc, 0.5)

    def test_summary(self):
        X, y = _make_regression(30, 2, 0.1)
        lr = LinearRegression()
        lr.fit(X, y)
        s = lr.summary()
        self.assertIn("LinearRegression", s)


class TestTreeModels(unittest.TestCase):

    def test_gini_impurity(self):
        g = gini_impurity([0, 0, 0])
        self.assertAlmostEqual(g, 0.0)
        g2 = gini_impurity([0, 1])
        self.assertAlmostEqual(g2, 0.5)

    def test_entropy(self):
        e = entropy([0, 0, 0])
        self.assertAlmostEqual(e, 0.0)
        e2 = entropy([0, 1])
        self.assertAlmostEqual(e2, 1.0)

    def test_mse(self):
        from intelligence.ml_basics.tree_models import mse as tree_mse
        v = tree_mse([1.0, 2.0, 3.0])
        self.assertAlmostEqual(v, 2.0 / 3.0)

    def test_decision_tree_classifier(self):
        X, y = _make_classification(80, 2)
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(X, y)
        preds = dt.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreaterEqual(acc, 0.6)

    def test_decision_tree_regressor(self):
        X, y = _make_regression(80, 3, 0.1)
        dt = DecisionTreeRegressor(max_depth=5)
        dt.fit(X, y)
        s = dt.score(X, y)
        self.assertGreater(s, 0.3)

    def test_random_forest_classifier(self):
        X, y = _make_classification(80, 2)
        rf = RandomForestClassifier(n_trees=10, max_depth=5)
        rf.fit(X, y)
        preds = rf.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreater(acc, 0.55)

    def test_random_forest_regressor(self):
        X, y = _make_regression(80, 3, 0.1)
        rf = RandomForestRegressor(n_trees=10, max_depth=5)
        rf.fit(X, y)
        s = rf.score(X, y)
        self.assertGreater(s, 0.3)

    def test_gradient_boosting_classifier(self):
        X, y = _make_classification(80, 2)
        gb = GradientBoostingClassifier(n_estimators=20, lr=0.1, max_depth=3)
        gb.fit(X, y)
        preds = gb.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreater(acc, 0.7)

    def test_gradient_boosting_regressor(self):
        X, y = _make_regression(80, 3, 0.1)
        gb = GradientBoostingRegressor(n_estimators=20, lr=0.1, max_depth=3)
        gb.fit(X, y)
        s = gb.score(X, y)
        self.assertGreater(s, 0.3)

    def test_feature_importances(self):
        X, y = _make_classification(80, 2)
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(X, y)
        self.assertEqual(len(dt.feature_importances_), 2)
        self.assertGreaterEqual(sum(dt.feature_importances_), 0.0)


class TestClustering(unittest.TestCase):

    def test_kmeans_fit_predict(self):
        X, y = _make_classification(60, 2)
        km = KMeans(k=2, max_iter=50)
        km.fit(X)
        self.assertEqual(len(km.labels), 60)
        self.assertIn(set(km.labels), [{0, 1}, {0}, {1}])

    def test_kmeans_inertia(self):
        X, y = _make_classification(60, 2)
        km = KMeans(k=2, max_iter=50)
        km.fit(X)
        self.assertGreaterEqual(km.inertia, 0.0)

    def test_kmeans_plus_plus(self):
        X, y = _make_classification(60, 2)
        km = KMeansPlusPlus(k=2, max_iter=50)
        km.fit(X)
        self.assertEqual(len(km.labels), 60)

    def test_dbscan(self):
        rng = random.Random(0)
        X = [[rng.gauss(0, 0.5) for _ in range(2)] for _ in range(30)] + [
            [rng.gauss(5, 0.5) for _ in range(2)] for _ in range(30)
        ]
        db = DBSCAN(eps=1.0, min_samples=3)
        db.fit(X)
        self.assertEqual(len(db.labels), 60)

    def test_hierarchical(self):
        X, y = _make_classification(30, 2)
        hc = HierarchicalClustering(linkage="single")
        hc.fit(X)
        labels = hc.cut(2)
        self.assertEqual(len(labels), 30)

    def test_agglomerative(self):
        X, y = _make_classification(30, 2)
        ac = AgglomerativeClustering(n_clusters=2)
        ac.fit(X)
        self.assertEqual(len(ac.labels), 30)

    def test_gmm(self):
        rng = random.Random(0)
        X = [[rng.gauss(0, 0.5) for _ in range(2)] for _ in range(30)] + [
            [rng.gauss(5, 0.5) for _ in range(2)] for _ in range(30)
        ]
        gmm = GMM(k=2, max_iter=20)
        gmm.fit(X)
        self.assertEqual(len(gmm.labels), 60)
        self.assertEqual(len(gmm.means), 2)

    def test_spectral_clustering(self):
        rng = random.Random(0)
        X = [[rng.gauss(0, 0.5) for _ in range(2)] for _ in range(20)] + [
            [rng.gauss(5, 0.5) for _ in range(2)] for _ in range(20)
        ]
        sc = SpectralClustering(n_clusters=2, gamma=1.0)
        sc.fit(X)
        self.assertEqual(len(sc.labels), 40)


class TestNeighbors(unittest.TestCase):

    def test_knn_classifier(self):
        X, y = _make_classification(60, 2)
        knn = KNeighborsClassifier(k=5)
        knn.fit(X, y)
        preds = knn.predict(X[:10])
        self.assertEqual(len(preds), 10)

    def test_knn_regressor(self):
        X, y = _make_regression(60, 3)
        knn = KNeighborsRegressor(k=5)
        knn.fit(X, y)
        preds = knn.predict(X[:10])
        self.assertEqual(len(preds), 10)

    def test_radius_neighbors(self):
        X, y = _make_classification(60, 2)
        rn = RadiusNeighborsClassifier(radius=2.0)
        rn.fit(X, y)
        preds = rn.predict(X[:10])
        self.assertEqual(len(preds), 10)

    def test_nearest_neighbors(self):
        X, y = _make_classification(60, 2)
        nn = NearestNeighbors()
        nn.fit(X)
        dists, indices = nn.kneighbors(X[0], k=3)
        self.assertEqual(len(dists), 3)
        self.assertEqual(len(indices), 3)

    def test_ball_tree(self):
        X, y = _make_classification(30, 2)
        bt = BallTree(X, leaf_size=5)
        results = bt.knn_query(X[0], k=3)
        self.assertEqual(len(results), 3)

    def test_euclidean_distance(self):
        d = euclidean_distance([0, 0], [3, 4])
        self.assertAlmostEqual(d, 5.0)

    def test_manhattan_distance(self):
        d = manhattan_distance([0, 0], [3, 4])
        self.assertAlmostEqual(d, 7.0)

    def test_cosine_distance(self):
        d = cosine_distance([1, 0], [1, 0])
        self.assertAlmostEqual(d, 0.0)

    def test_knn_distance_weights(self):
        X, y = _make_classification(60, 2)
        knn = KNeighborsClassifier(k=5, weights="distance")
        knn.fit(X, y)
        preds = knn.predict(X[:10])
        self.assertEqual(len(preds), 10)


class TestPreprocessing(unittest.TestCase):

    def test_min_max_scaler(self):
        X, y = _make_regression(30, 3)
        scaler = MinMaxScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        for row in Xt:
            for v in row:
                self.assertGreaterEqual(v, -0.01)
                self.assertLessEqual(v, 1.01)

    def test_standard_scaler(self):
        X, y = _make_regression(30, 3)
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        means = [sum(row[j] for row in Xt) / len(Xt) for j in range(3)]
        for m in means:
            self.assertAlmostEqual(m, 0.0, delta=0.2)

    def test_robust_scaler(self):
        X, y = _make_regression(30, 3)
        scaler = RobustScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        self.assertEqual(len(Xt), 30)

    def test_normalizer(self):
        X = [[3.0, 4.0], [1.0, 0.0]]
        norm = Normalizer(norm="l2")
        Xt = norm.transform(X)
        self.assertAlmostEqual(math.sqrt(Xt[0][0] ** 2 + Xt[0][1] ** 2), 1.0)

    def test_one_hot_encoder(self):
        X = [["red"], ["blue"], ["green"], ["red"]]
        enc = OneHotEncoder()
        enc.fit(X)
        Xt = enc.transform(X)
        self.assertEqual(len(Xt[0]), 3)

    def test_label_encoder(self):
        y = ["cat", "dog", "cat", "fish"]
        le = LabelEncoder()
        le.fit(y)
        yt = le.transform(y)
        self.assertEqual(yt, [0, 1, 0, 2])

    def test_polynomial_features(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        pf = PolynomialFeatures(degree=2)
        pf.fit(X)
        Xt = pf.transform(X)
        self.assertEqual(len(Xt[0]), 5)

    def test_train_test_split(self):
        X, y = _make_regression(100, 3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

    def test_kfold(self):
        X, y = _make_regression(100, 3)
        kf = KFold(n_splits=5)
        splits = kf.split(X)
        self.assertEqual(len(splits), 5)

    def test_stratified_kfold(self):
        X, y = _make_classification(100, 2)
        skf = StratifiedKFold(n_splits=5)
        splits = skf.split(X, y)
        self.assertEqual(len(splits), 5)
        for train_idx, test_idx in splits:
            self.assertEqual(len(train_idx) + len(test_idx), 100)

    def test_cross_val_score(self):
        X, y = _make_classification(60, 2)
        lr = LogisticRegression(lr=0.5, epochs=50)
        scores = cross_val_score(lr, X, y, KFold(n_splits=3))
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)

    def test_inverse_transform(self):
        X, y = _make_regression(30, 3)
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        X_inv = scaler.inverse_transform(Xt)
        for i in range(30):
            for j in range(3):
                self.assertAlmostEqual(X[i][j], X_inv[i][j], delta=1e-6)


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.6)

    def test_precision(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        p = precision(y_true, y_pred, average="macro")
        self.assertGreater(p, 0.0)

    def test_recall(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        r = recall(y_true, y_pred, average="macro")
        self.assertGreater(r, 0.0)

    def test_f1(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        f = f1_score(y_true, y_pred, average="macro")
        self.assertGreater(f, 0.0)

    def test_confusion_matrix(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm, [[1, 1], [1, 1]])

    def test_classification_report(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        report = classification_report(y_true, y_pred)
        self.assertIn("precision", report)

    def test_roc_auc(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.6, 0.9]
        auc = roc_auc_score(y_true, y_scores)
        self.assertGreater(auc, 0.5)

    def test_mse(self):
        self.assertAlmostEqual(mse([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(mse([1, 2], [0, 0]), 2.5)

    def test_rmse(self):
        self.assertAlmostEqual(rmse([1, 2, 3], [1, 2, 3]), 0.0)

    def test_mae(self):
        self.assertAlmostEqual(mae([1, 2, 3], [2, 3, 4]), 1.0)

    def test_r2(self):
        self.assertAlmostEqual(r2_score([1, 2, 3], [1, 2, 3]), 1.0)

    def test_adjusted_r2(self):
        ar2 = adjusted_r2([1, 2, 3, 4], [1, 2, 3, 4], n_features=1)
        self.assertGreaterEqual(ar2, 0.0)

    def test_silhouette(self):
        X = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]]
        labels = [0, 0, 0, 1, 1, 1]
        s = silhouette_score(X, labels)
        self.assertGreater(s, 0.5)

    def test_davies_bouldin(self):
        X = [[0, 0], [0, 1], [5, 5], [5, 6]]
        labels = [0, 0, 1, 1]
        db = davies_bouldin_score(X, labels)
        self.assertGreater(db, 0.0)

    def test_calinski_harabasz(self):
        X = [[0, 0], [0, 1], [5, 5], [5, 6]]
        labels = [0, 0, 1, 1]
        ch = calinski_harabasz_score(X, labels)
        self.assertGreater(ch, 0.0)

    def test_ndcg(self):
        y_true = [1, 0, 1, 0]
        y_scores = [0.9, 0.8, 0.3, 0.1]
        n = ndcg(y_true, y_scores, k=4)
        self.assertGreaterEqual(n, 0.0)

    def test_map_score(self):
        y_true = [1, 0, 1, 0]
        y_scores = [0.9, 0.8, 0.3, 0.1]
        m = map_score(y_true, y_scores, k=4)
        self.assertGreaterEqual(m, 0.0)


class TestEnsemble(unittest.TestCase):

    def test_voting_classifier(self):
        X, y = _make_classification(60, 2)
        vc = VotingClassifier(
            estimators=[
                ("lr1", LogisticRegression(lr=0.5, epochs=100)),
                ("lr2", LogisticRegression(lr=0.3, epochs=100)),
            ],
            strategy="hard",
        )
        vc.fit(X, y)
        preds = vc.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreater(acc, 0.5)

    def test_bagging_classifier(self):
        X, y = _make_classification(60, 2)
        bc = BaggingClassifier(n_estimators=5)
        bc.fit(X, y)
        preds = bc.predict(X)
        self.assertEqual(len(preds), 60)

    def test_ada_boost_classifier(self):
        X, y = _make_classification(60, 2)
        ab = AdaBoostClassifier(n_estimators=10, lr=0.5)
        ab.fit(X, y)
        preds = ab.predict(X)
        acc = sum(1 for a, b in zip(y, preds) if a == b) / len(y)
        self.assertGreater(acc, 0.5)

    def test_ada_boost_regressor(self):
        X, y = _make_regression(60, 3, 0.1)
        ab = AdaBoostRegressor(n_estimators=10, lr=0.5)
        ab.fit(X, y)
        preds = ab.predict(X)
        self.assertEqual(len(preds), 60)


class TestFeatureSelection(unittest.TestCase):

    def test_variance_threshold(self):
        X = [[1, 0, 5], [1, 0, 5], [1, 0, 5], [2, 1, 6]]
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X)
        Xt = vt.transform(X)
        self.assertLessEqual(len(Xt[0]), 3)

    def test_correlation_filter(self):
        X = [[1, 1, 2], [2, 2, 4], [3, 3, 6], [4, 4, 8]]
        cf = CorrelationFilter(threshold=0.9)
        cf.fit(X)
        Xt = cf.transform(X)
        self.assertLessEqual(len(Xt[0]), 3)

    def test_mutual_information(self):
        X, y = _make_classification(50, 3)
        mi = mutual_information(X, y)
        self.assertEqual(len(mi), 3)
        for v in mi:
            self.assertGreaterEqual(v, 0.0)

    def test_chi_squared(self):
        X, y = _make_classification(50, 3)
        cs = chi_squared(X, y)
        self.assertEqual(len(cs), 3)

    def test_select_k_best(self):
        X, y = _make_classification(50, 3)
        skb = SelectKBest(k=2)
        skb.fit(X, y)
        Xt = skb.transform(X)
        self.assertEqual(len(Xt[0]), 2)


class TestGetParamsAndSummary(unittest.TestCase):

    def test_logistic_get_params(self):
        lr = LogisticRegression(lr=0.5, epochs=100)
        params = lr.get_params()
        self.assertEqual(params["lr"], 0.5)

    def test_kmeans_get_params(self):
        km = KMeans(k=3)
        km.fit([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]])
        params = km.get_params()
        self.assertEqual(params["k"], 3)

    def test_dbscan_get_params(self):
        db = DBSCAN(eps=0.5, min_samples=3)
        db.fit([[0, 0], [0, 1], [1, 0], [5, 5]])
        params = db.get_params()
        self.assertEqual(params["eps"], 0.5)


if __name__ == "__main__":
    unittest.main()
