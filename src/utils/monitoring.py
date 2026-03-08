from __future__ import annotations

from typing import Sequence

import numpy as np


def entropy(proba: np.ndarray, eps: float = 1.0e-10) -> float:
    p = np.asarray(proba, dtype=np.float64).ravel()
    return float(-np.sum(p * np.log(p + eps)))


def confidence(proba: np.ndarray) -> float:
    p = np.asarray(proba, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    return float(np.max(p))


def class_kl_div(pred_labels: Sequence[int], n_classes: int, eps: float = 1.0e-10) -> float:
    labels = np.asarray(list(pred_labels), dtype=np.int64)
    if labels.size == 0:
        return 0.0

    counts = np.bincount(labels, minlength=int(n_classes)).astype(np.float64)
    dist = counts / float(labels.size)
    uniform = np.ones(int(n_classes), dtype=np.float64) / float(n_classes)
    return float(np.sum(dist * np.log((dist + eps) / (uniform + eps))))

