from __future__ import annotations

from typing import Optional

import numpy as np


def matrix_power_spd(C: np.ndarray, power: float, eps: float = 1.0e-6) -> np.ndarray:
    """Compute SPD matrix power via eigen-decomposition."""
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, eps, None)
    eigvals_p = eigvals ** power
    return (eigvecs * eigvals_p[None, :]) @ eigvecs.T


def compute_alignment_matrix(C_source: np.ndarray, C_target: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """A = sqrt(C_source) @ inv_sqrt(C_target)."""
    Cs_half = matrix_power_spd(C_source, 0.5, eps=eps)
    Ct_mhalf = matrix_power_spd(C_target, -0.5, eps=eps)
    return Cs_half @ Ct_mhalf


def apply_alignment(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Left-multiply each trial by A: (n_trials, C, T) -> (n_trials, C, T)."""
    return np.einsum("ij,tjs->tis", A, X)


class EuclideanAlignment:
    def __init__(self, eps: float = 1.0e-6):
        self.eps = float(eps)
        self.C_ref: Optional[np.ndarray] = None

    def fit(self, C_source: np.ndarray):  # noqa: N803
        self.C_ref = np.mean(C_source, axis=0)
        return self

    def compute_matrix(self, C_target: np.ndarray) -> np.ndarray:  # noqa: N803
        if self.C_ref is None:
            raise RuntimeError("EuclideanAlignment is not fitted.")
        C_target_mean = np.mean(C_target, axis=0)
        return compute_alignment_matrix(self.C_ref, C_target_mean, eps=self.eps)
