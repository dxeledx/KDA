from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def cka(X: np.ndarray, Y: np.ndarray) -> float:  # noqa: N803
    """Linear CKA with Gram matrix centering."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"CKA expects same n_samples: {X.shape} vs {Y.shape}")

    K = X @ X.T
    L = Y @ Y.T

    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    numerator = np.sum(Kc * Lc)
    denominator = np.linalg.norm(Kc, "fro") * np.linalg.norm(Lc, "fro")
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)

