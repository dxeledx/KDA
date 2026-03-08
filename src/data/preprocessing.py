from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    """Lightweight preprocessing on already-epoched numpy arrays.

    MOABB handles filtering and epoching. This class optionally applies
    per-trial, per-channel z-score normalization.
    """

    def __init__(self, normalize: bool = False, eps: float = 1.0e-8):
        self.normalize = normalize
        self.eps = eps

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):  # noqa: N803
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        X = np.asarray(X)
        if not self.normalize:
            return X

        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True) + self.eps
        return (X - mean) / std
