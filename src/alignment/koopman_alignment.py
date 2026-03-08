from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import eigh
from sklearn.decomposition import PCA

from src.alignment.euclidean import matrix_power_spd


def _sym_to_vec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    values = []
    sqrt2 = float(np.sqrt(2.0))
    for row in range(n):
        values.append(float(matrix[row, row]))
        for col in range(row + 1, n):
            values.append(sqrt2 * float(matrix[row, col]))
    return np.asarray(values, dtype=np.float64)


def _matrix_log_spd(matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    eigvals, eigvecs = eigh(0.5 * (matrix + matrix.T))
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def _lift_quadratic(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    return np.concatenate([states, states * states, np.ones((states.shape[0], 1))], axis=1)


@dataclass
class KoopmanFeatureProjector:
    pca_rank: int = 16
    cov_eps: float = 1.0e-6
    reference_covariance: np.ndarray | None = None
    pca: PCA | None = None

    def fit(self, covariances: np.ndarray) -> "KoopmanFeatureProjector":
        covariances = np.asarray(covariances, dtype=np.float64)
        if self.reference_covariance is None:
            self.reference_covariance = mean_riemann(covariances)
        tangent = self._tangent_vectors(covariances)
        if tangent.shape[1] > self.pca_rank:
            self.pca = PCA(
                n_components=min(int(self.pca_rank), tangent.shape[0], tangent.shape[1])
            )
            self.pca.fit(tangent)
        else:
            self.pca = None
        return self

    def _tangent_vectors(self, covariances: np.ndarray) -> np.ndarray:
        if self.reference_covariance is None:
            raise RuntimeError("KoopmanFeatureProjector is not fitted.")
        ref_inv_sqrt = matrix_power_spd(self.reference_covariance, -0.5, eps=self.cov_eps)
        vectors = []
        for covariance in np.asarray(covariances, dtype=np.float64):
            whitened = ref_inv_sqrt @ covariance @ ref_inv_sqrt
            tangent = _matrix_log_spd(whitened)
            vectors.append(_sym_to_vec(tangent))
        return np.asarray(vectors, dtype=np.float64)

    def transform_tangent(self, covariances: np.ndarray) -> np.ndarray:
        tangent = self._tangent_vectors(covariances)
        if self.pca is not None:
            return self.pca.transform(tangent)
        return tangent

    def transform(self, covariances: np.ndarray) -> np.ndarray:
        return _lift_quadratic(self.transform_tangent(covariances))


@dataclass
class KoopmanAffineAligner:
    eps: float = 1.0e-6
    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    matrix_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "KoopmanAffineAligner":
        source_features = np.asarray(source_features, dtype=np.float64)
        target_features = np.asarray(target_features, dtype=np.float64)
        self.source_mean_ = source_features.mean(axis=0)
        self.target_mean_ = target_features.mean(axis=0)
        source_cov = np.cov(source_features, rowvar=False)
        target_cov = np.cov(target_features, rowvar=False)
        source_half = matrix_power_spd(source_cov, 0.5, eps=self.eps)
        target_inv_half = matrix_power_spd(target_cov, -0.5, eps=self.eps)
        self.matrix_ = source_half @ target_inv_half
        return self

    def transform(self, target_features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None or self.matrix_ is None:
            raise RuntimeError("KoopmanAffineAligner is not fitted.")
        target_features = np.asarray(target_features, dtype=np.float64)
        centered = target_features - self.target_mean_
        return centered @ self.matrix_.T + self.source_mean_


def fit_alignment(
    source_covariances: np.ndarray,
    target_covariances: np.ndarray,
    pca_rank: int = 16,
    cov_eps: float = 1.0e-6,
) -> tuple[KoopmanFeatureProjector, KoopmanAffineAligner, np.ndarray, np.ndarray]:
    projector = KoopmanFeatureProjector(pca_rank=pca_rank, cov_eps=cov_eps).fit(
        source_covariances
    )
    psi_source = projector.transform(source_covariances)
    psi_target = projector.transform(target_covariances)
    aligner = KoopmanAffineAligner(eps=cov_eps).fit(psi_source, psi_target)
    return projector, aligner, psi_source, psi_target


def transform(
    covariances: np.ndarray,
    projector: KoopmanFeatureProjector,
    aligner: KoopmanAffineAligner | None = None,
) -> np.ndarray:
    psi = projector.transform(covariances)
    return psi if aligner is None else aligner.transform(psi)
