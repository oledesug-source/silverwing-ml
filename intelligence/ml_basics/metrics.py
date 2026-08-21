"""Evaluation metrics for classification, regression, clustering, and ranking."""

from __future__ import annotations

import math

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "classification_report",
    "roc_auc_score",
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "adjusted_r2",
    "mape",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "ndcg",
    "map_score",
]


def accuracy(y_true: list, y_pred: list) -> float:
    """Return fraction of correct predictions."""
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _per_class_counts(y_true: list, y_pred: list, labels: list) -> dict:
    tp = dict.fromkeys(labels, 0)
    fp = dict.fromkeys(labels, 0)
    fn = dict.fromkeys(labels, 0)
    for t, p in zip(y_true, y_pred):
        for l in labels:
            if t == l and p == l:
                tp[l] += 1
            elif t != l and p == l:
                fp[l] += 1
            elif t == l and p != l:
                fn[l] += 1
    return tp, fp, fn


def precision(y_true: list, y_pred: list, average: str = "macro") -> float:
    """Compute precision score."""
    labels = sorted(set(y_true) | set(y_pred))
    tp, fp, fn = _per_class_counts(y_true, y_pred, labels)
    if average == "macro":
        vals = []
        for l in labels:
            denom = tp[l] + fp[l]
            vals.append(tp[l] / denom if denom > 0 else 0.0)
        return sum(vals) / len(vals) if vals else 0.0
    elif average == "micro":
        t = sum(tp[l] for l in labels)
        f = sum(fp[l] for l in labels)
        return t / (t + f) if (t + f) > 0 else 0.0
    elif average == "weighted":
        vals = []
        for l in labels:
            denom = tp[l] + fp[l]
            p = tp[l] / denom if denom > 0 else 0.0
            support = sum(1 for t in y_true if t == l)
            vals.append(p * support)
        total = len(y_true)
        return sum(vals) / total if total > 0 else 0.0
    else:
        raise ValueError(f"Unknown average: {average}")


def recall(y_true: list, y_pred: list, average: str = "macro") -> float:
    """Compute recall score."""
    labels = sorted(set(y_true) | set(y_pred))
    tp, fp, fn = _per_class_counts(y_true, y_pred, labels)
    if average == "macro":
        vals = []
        for l in labels:
            denom = tp[l] + fn[l]
            vals.append(tp[l] / denom if denom > 0 else 0.0)
        return sum(vals) / len(vals) if vals else 0.0
    elif average == "micro":
        t = sum(tp[l] for l in labels)
        f = sum(fn[l] for l in labels)
        return t / (t + f) if (t + f) > 0 else 0.0
    elif average == "weighted":
        vals = []
        for l in labels:
            denom = tp[l] + fn[l]
            r = tp[l] / denom if denom > 0 else 0.0
            support = sum(1 for t in y_true if t == l)
            vals.append(r * support)
        total = len(y_true)
        return sum(vals) / total if total > 0 else 0.0
    else:
        raise ValueError(f"Unknown average: {average}")


def f1_score(y_true: list, y_pred: list, average: str = "macro") -> float:
    """Compute F1 score."""
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def confusion_matrix(y_true: list, y_pred: list) -> list[list[int]]:
    """Return confusion matrix as list of lists."""
    labels = sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        mat[idx[t]][idx[p]] += 1
    return mat


def classification_report(y_true: list, y_pred: list) -> str:
    """Return a formatted classification report string."""
    labels = sorted(set(y_true) | set(y_pred))
    tp, fp, fn = _per_class_counts(y_true, y_pred, labels)
    lines = ["", "              precision    recall  f1-score   support"]
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    total_support = 0
    for l in labels:
        support = sum(1 for t in y_true if t == l)
        p_val = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        r_val = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        macro_p += p_val
        macro_r += r_val
        macro_f1 += f1_val
        total_support += support
        lines.append(f"        {l:>6}    {p_val:.2f}      {r_val:.2f}      {f1_val:.2f}      {support:>6}")
    n = len(labels)
    macro_p /= n if n else 1
    macro_r /= n if n else 1
    macro_f1 /= n if n else 1
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0.0
    lines.append("")
    lines.append(f"    accuracy                          {acc:.2f}      {total_support:>6}")
    lines.append(f"   macro avg     {macro_p:.2f}      {macro_r:.2f}      {macro_f1:.2f}      {total_support:>6}")
    lines.append(f"weighted avg     {macro_p:.2f}      {macro_r:.2f}      {macro_f1:.2f}      {total_support:>6}")
    lines.append("")
    return "\n".join(lines)


def roc_auc_score(y_true: list, y_scores: list) -> float:
    """Compute ROC AUC score for binary classification."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tp = 0
    fp = 0
    tpr_list = [0.0]
    fpr_list = [0.0]
    for _score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0
    return auc


def mse(y_true: list, y_pred: list) -> float:
    """Compute mean squared error."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n


def rmse(y_true: list, y_pred: list) -> float:
    """Compute root mean squared error."""
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: list, y_pred: list) -> float:
    """Compute mean absolute error."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n


def r2_score(y_true: list, y_pred: list) -> float:
    """Compute R-squared score."""
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_y = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def adjusted_r2(y_true: list, y_pred: list, n_features: int) -> float:
    """Compute adjusted R-squared."""
    n = len(y_true)
    if n == 0 or n <= n_features + 1:
        return 0.0
    r2 = r2_score(y_true, y_pred)
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)


def mape(y_true: list, y_pred: list) -> float:
    """Compute mean absolute percentage error."""
    n = len(y_true)
    if n == 0:
        return 0.0
    total = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        if t != 0:
            total += abs((t - p) / t)
            count += 1
    return (total / count * 100.0) if count > 0 else 0.0


def silhouette_score(X: list[list[float]], labels: list[int]) -> float:
    """Compute mean silhouette coefficient over all samples."""
    n = len(X)
    if n <= 1:
        return 0.0
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 1:
        return 0.0

    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    total = 0.0
    for i in range(n):
        own = labels[i]
        same_cluster = [j for j in range(n) if j != i and labels[j] == own]
        if same_cluster:
            a_i = sum(_dist(X[i], X[j]) for j in same_cluster) / len(same_cluster)
        else:
            a_i = 0.0
        b_i = float("inf")
        for lab in unique_labels:
            if lab == own:
                continue
            others = [j for j in range(n) if labels[j] == lab]
            if others:
                d = sum(_dist(X[i], X[j]) for j in others) / len(others)
                if d < b_i:
                    b_i = d
        if b_i == float("inf"):
            b_i = 0.0
        denom = max(a_i, b_i)
        total += (b_i - a_i) / denom if denom > 0 else 0.0
    return total / n


def davies_bouldin_score(X: list[list[float]], labels: list[int]) -> float:
    """Compute Davies-Bouldin index."""
    n = len(X)
    if n <= 1:
        return 0.0
    unique_labels = sorted(set(labels))

    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    centroids = []
    for lab in unique_labels:
        members = [X[i] for i in range(n) if labels[i] == lab]
        if members:
            dim = len(members[0])
            c = [sum(m[j] for m in members) / len(members) for j in range(dim)]
        else:
            c = [0.0] * len(X[0])
        centroids.append(c)

    dispersions = []
    for idx_l, lab in enumerate(unique_labels):
        members = [X[i] for i in range(n) if labels[i] == lab]
        if members:
            avg_d = sum(_dist(m, centroids[idx_l]) for m in members) / len(members)
        else:
            avg_d = 0.0
        dispersions.append(avg_d)

    db_values = []
    for i in range(len(unique_labels)):
        max_r = 0.0
        for j in range(len(unique_labels)):
            if i == j:
                continue
            denom = dispersions[i] + dispersions[j]
            r = _dist(centroids[i], centroids[j]) / denom if denom > 0 else 0.0
            if r > max_r:
                max_r = r
        db_values.append(max_r)
    return sum(db_values) / len(db_values) if db_values else 0.0


def calinski_harabasz_score(X: list[list[float]], labels: list[int]) -> float:
    """Compute Calinski-Harabasz index."""
    n = len(X)
    if n <= 1:
        return 0.0
    unique_labels = sorted(set(labels))
    k = len(unique_labels)
    if k <= 1:
        return 0.0
    dim = len(X[0])
    overall_centroid = [sum(X[i][j] for i in range(n)) / n for j in range(dim)]
    ssw = 0.0
    ssb = 0.0
    for lab in unique_labels:
        members = [X[i] for i in range(n) if labels[i] == lab]
        nc = len(members)
        if nc == 0:
            continue
        centroid = [sum(m[j] for m in members) / nc for j in range(dim)]
        for m in members:
            ssw += sum((m[j] - centroid[j]) ** 2 for j in range(dim))
        ssb += nc * sum((centroid[j] - overall_centroid[j]) ** 2 for j in range(dim))
    if ssw == 0:
        return 0.0
    return (ssb / (k - 1)) / (ssw / (n - k))


def ndcg(y_true: list, y_scores: list, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    def _dcg(relevances: list[float], cap: int) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:cap]))

    paired = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    pred_rel = [p for _, p in paired[:k]]
    ideal_rel = sorted(y_true, reverse=True)[:k]
    dcg_val = _dcg(pred_rel, k)
    idcg_val = _dcg(ideal_rel, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def map_score(y_true: list, y_scores: list, k: int = 10) -> float:
    """Compute Mean Average Precision at k."""
    paired = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    top_k = [label for _, label in paired[:k]]
    relevant_count = sum(1 for l in y_true if l == 1)
    if relevant_count == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, label in enumerate(top_k):
        if label == 1:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / relevant_count if relevant_count > 0 else 0.0
