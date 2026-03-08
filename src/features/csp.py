from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.covariance import compute_covariances, mean_covariance


class CSP(BaseEstimator, TransformerMixin):
    """Multi-class CSP using One-vs-Rest strategy.

    For K classes, trains K binary CSP filters. Each binary CSP yields
    `n_components` features (typically 6: 3 smallest + 3 largest eigenvectors),
    leading to `K * n_components` output features.
    """

    def __init__(
        self,
        n_components: int = 6,
        reg: float = 0.1,
        eps: float = 1.0e-6,
        log: bool = True,
    ):
        self.n_components = int(n_components)
        self.reg = float(reg)
        self.eps = float(eps)
        self.log = bool(log)

        self.classes_: Optional[np.ndarray] = None
        self.filters_: Optional[np.ndarray] = None  # (n_classes, n_components, n_channels)

    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: N803
        X = np.asarray(X)
        y = np.asarray(y)
        covs = compute_covariances(X, eps=self.eps)

        classes = np.unique(y)
        self.classes_ = classes

        n_channels = X.shape[1]
        ident = np.eye(n_channels, dtype=np.float64)

        filters = []
        n_half = self.n_components // 2
        if 2 * n_half != self.n_components:
            raise ValueError("n_components must be even (e.g., 6).")

        for cls in classes:
            cov_cls = mean_covariance(covs[y == cls])
            cov_rest = mean_covariance(covs[y != cls])

            # Regularization (shrink towards identity)
            cov_cls = (1.0 - self.reg) * cov_cls + self.reg * ident / n_channels
            cov_rest = (1.0 - self.reg) * cov_rest + self.reg * ident / n_channels

            cov_sum = cov_cls + cov_rest

            evals, evecs = eigh(cov_cls, cov_sum)
            order = np.argsort(evals)  # ascending
            pick = np.concatenate([order[:n_half], order[-n_half:]])
            W = evecs[:, pick].T  # (n_components, n_channels)
            filters.append(W)

        self.filters_ = np.stack(filters, axis=0).astype(np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        if self.filters_ is None or self.classes_ is None:
            raise RuntimeError("CSP is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        covariances = compute_covariances(X, eps=self.eps)
        return self.transform_covariances(covariances)

    def transform_covariances(self, covariances: np.ndarray) -> np.ndarray:
        if self.filters_ is None or self.classes_ is None:
            raise RuntimeError("CSP is not fitted.")

        covariances = np.asarray(covariances, dtype=np.float64)
        features_per_class = []
        for W in self.filters_:
            proj_cov = np.einsum("fc,tcd,gd->tfg", W, covariances, W)  # (n_trials, n_components, n_components)
            var = np.clip(np.diagonal(proj_cov, axis1=1, axis2=2), 1.0e-12, None)
            if self.log:
                var = np.log(var)
            features_per_class.append(var)

        return np.concatenate(features_per_class, axis=1).astype(np.float64)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.fit(X, y).transform(X)
