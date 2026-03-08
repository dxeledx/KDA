from __future__ import annotations

import numpy as np


def compute_covariances(X: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:  # noqa: N803
    """Compute per-trial normalized covariance matrices.

    Args:
        X: (n_trials, n_channels, n_samples)
        eps: Diagonal regularization to ensure SPD

    Returns:
        covs: (n_trials, n_channels, n_channels)
    """
    X = np.asarray(X, dtype=np.float64)
    covs = np.einsum("tci,tdi->tcd", X, X)
    traces = np.trace(covs, axis1=1, axis2=2)
    covs = covs / traces[:, None, None]
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
    if eps > 0:
        n_channels = covs.shape[1]
        covs = covs + eps * np.eye(n_channels)[None, :, :]
    return covs.astype(np.float64)


def mean_covariance(covs: np.ndarray) -> np.ndarray:
    covs = np.asarray(covs, dtype=np.float64)
    C = covs.mean(axis=0)
    return 0.5 * (C + C.T)

