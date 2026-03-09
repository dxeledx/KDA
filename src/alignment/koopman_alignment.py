from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import eigh
from scipy.optimize import minimize
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


def _lift_quadratic_cubic(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    return np.concatenate(
        [states, states * states, states * states * states, np.ones((states.shape[0], 1))],
        axis=1,
    )


def _apply_lifting(states: np.ndarray, lifting: str) -> np.ndarray:
    if lifting == "quadratic":
        return _lift_quadratic(states)
    if lifting == "quadratic_cubic":
        return _lift_quadratic_cubic(states)
    raise ValueError(f"Unsupported lifting '{lifting}'")


def _one_hot(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    eye = np.eye(len(classes), dtype=np.float64)
    mapping = {int(label): idx for idx, label in enumerate(classes)}
    return np.asarray([eye[mapping[int(label)]] for label in y], dtype=np.float64)


def _safe_cov(X: np.ndarray, reg_lambda: float = 0.0) -> np.ndarray:  # noqa: N803
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {X.shape}")
    if X.shape[0] <= 1:
        return np.eye(X.shape[1], dtype=np.float64) * max(float(reg_lambda), 1.0e-6)
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    if reg_lambda > 0.0:
        cov = cov + float(reg_lambda) * np.eye(cov.shape[0], dtype=np.float64)
    return np.asarray(cov, dtype=np.float64)


def _orthonormalize(columns: np.ndarray) -> np.ndarray:
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {columns.shape}")
    if columns.size == 0:
        return np.zeros((columns.shape[0], 0), dtype=np.float64)
    q, _ = np.linalg.qr(columns)
    keep = np.linalg.norm(q, axis=0) > 1.0e-8
    if not np.any(keep):
        return np.zeros((columns.shape[0], 0), dtype=np.float64)
    return np.asarray(q[:, keep], dtype=np.float64)


def _complete_basis(primary: np.ndarray, X_centered: np.ndarray, k: int) -> np.ndarray:
    X_centered = np.asarray(X_centered, dtype=np.float64)
    d = X_centered.shape[1]
    k = int(max(1, min(k, d)))
    basis = _orthonormalize(primary)

    if basis.shape[1] < k:
        if basis.shape[1] > 0:
            residual = X_centered - X_centered @ basis @ basis.T
        else:
            residual = X_centered
        if residual.size > 0:
            _, _, vh = np.linalg.svd(residual, full_matrices=False)
            extra = _orthonormalize(vh.T)
            if basis.shape[1] > 0 and extra.shape[1] > 0:
                extra = extra - basis @ (basis.T @ extra)
                extra = _orthonormalize(extra)
            if extra.shape[1] > 0:
                basis = np.concatenate([basis, extra[:, : max(0, k - basis.shape[1])]], axis=1)

    if basis.shape[1] < k:
        canonical = np.eye(d, dtype=np.float64)
        if basis.shape[1] > 0:
            canonical = canonical - basis @ (basis.T @ canonical)
            canonical = _orthonormalize(canonical)
        basis = np.concatenate([basis, canonical[:, : max(0, k - basis.shape[1])]], axis=1)

    return np.asarray(basis[:, :k], dtype=np.float64)


def _fit_lda_directions(X_centered: np.ndarray, y: np.ndarray, reg_lambda: float) -> np.ndarray:  # noqa: N803
    classes = np.unique(y)
    global_mean = X_centered.mean(axis=0)
    within = np.zeros((X_centered.shape[1], X_centered.shape[1]), dtype=np.float64)
    between = np.zeros_like(within)
    for label in classes:
        block = X_centered[y == label]
        if len(block) == 0:
            continue
        class_mean = block.mean(axis=0)
        diff = class_mean - global_mean
        within += block.shape[0] * _safe_cov(block - class_mean, reg_lambda=0.0)
        between += block.shape[0] * np.outer(diff, diff)
    within += float(reg_lambda) * np.eye(within.shape[0], dtype=np.float64)
    eigvals, eigvecs = eigh(between, within)
    order = np.argsort(eigvals)[::-1]
    return np.asarray(eigvecs[:, order], dtype=np.float64)


def _fit_csp_directions(X_centered: np.ndarray, y: np.ndarray, reg_lambda: float) -> np.ndarray:  # noqa: N803
    classes = np.unique(y)
    candidates = []
    scores = []
    for label in classes:
        pos = X_centered[y == label]
        neg = X_centered[y != label]
        if len(pos) <= 1 or len(neg) <= 1:
            continue
        cov_pos = _safe_cov(pos, reg_lambda=reg_lambda)
        cov_neg = _safe_cov(neg, reg_lambda=reg_lambda)
        composite = cov_pos + cov_neg + float(reg_lambda) * np.eye(cov_pos.shape[0], dtype=np.float64)
        eigvals, eigvecs = eigh(cov_pos, composite)
        order = np.argsort(np.abs(eigvals - 0.5))[::-1]
        candidates.append(eigvecs[:, order])
        scores.extend(np.abs(eigvals[order] - 0.5).tolist())
    if not candidates:
        return np.zeros((X_centered.shape[1], 0), dtype=np.float64)
    stacked = np.concatenate(candidates, axis=1)
    score_order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    return np.asarray(stacked[:, score_order], dtype=np.float64)


def _fit_linear_directions(X_centered: np.ndarray, y: np.ndarray, reg_lambda: float) -> np.ndarray:  # noqa: N803
    targets = _one_hot(y)
    gram = X_centered.T @ X_centered + float(reg_lambda) * np.eye(X_centered.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, X_centered.T @ targets)
    return np.asarray(weights, dtype=np.float64)


def _split_blocks(states: np.ndarray, block_lengths: list[int] | tuple[int, ...] | np.ndarray) -> list[np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    lengths = [int(length) for length in block_lengths]
    if not lengths:
        raise ValueError("source_block_lengths must not be empty.")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"Invalid source_block_lengths: {lengths}")
    if int(sum(lengths)) != int(states.shape[0]):
        raise ValueError(
            f"source_block_lengths must sum to {states.shape[0]}, got {sum(lengths)}"
        )

    blocks = []
    start = 0
    for length in lengths:
        blocks.append(states[start : start + length])
        start += length
    return blocks


def _fit_linear_state_operator_blocks(
    state_blocks: list[np.ndarray],
    ridge_alpha: float = 1.0e-3,
) -> np.ndarray:
    prev_blocks, next_blocks = [], []
    for block in state_blocks:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] < 2:
            continue
        prev_blocks.append(block[:-1])
        next_blocks.append(block[1:])

    if not prev_blocks:
        raise ValueError("Need at least one valid source block with two states.")

    prev = np.concatenate(prev_blocks, axis=0)
    nxt = np.concatenate(next_blocks, axis=0)
    gram = prev.T @ prev + float(ridge_alpha) * np.eye(prev.shape[1], dtype=np.float64)
    operator = (nxt.T @ prev) @ np.linalg.pinv(gram)
    return np.asarray(operator, dtype=np.float64)


def _decision_scores_from_lda(lda, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(lda.model.decision_function(np.asarray(features, dtype=np.float64)), dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores


@dataclass
class KoopmanFeatureProjector:
    pca_rank: int = 16
    cov_eps: float = 1.0e-6
    lifting: str = "quadratic"
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
        return _apply_lifting(self.transform_tangent(covariances), self.lifting)


@dataclass
class KoopmanConservativeResidualAligner:
    residual_rank: int = 8
    basis_k: int = 32
    basis_reg_lambda: float = 1.0e-4
    lambda_cls: float = 1.0
    lambda_dyn: float = 1.0
    lambda_reg: float = 1.0e-3
    ridge_alpha: float = 1.0e-3
    max_iter: int = 100
    tol: float = 1.0e-7
    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    basis_: np.ndarray | None = None
    residual_matrix_: np.ndarray | None = None
    source_operator_: np.ndarray | None = None
    optimization_info_: dict | None = None

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        *,
        y_source: np.ndarray,
        source_block_lengths: list[int] | tuple[int, ...] | np.ndarray | None = None,
    ) -> "KoopmanConservativeResidualAligner":
        source = np.asarray(source_features, dtype=np.float64)
        target = np.asarray(target_features, dtype=np.float64)
        y_source = np.asarray(y_source, dtype=np.int64)

        if source.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Expected 2D feature matrices, got {source.shape} and {target.shape}")
        if source.shape[1] != target.shape[1]:
            raise ValueError(f"Feature dimension mismatch: {source.shape} vs {target.shape}")
        if len(source) != len(y_source):
            raise ValueError(f"y_source length mismatch: {len(source)} vs {len(y_source)}")
        if float(self.lambda_dyn) > 0.0 and source_block_lengths is None:
            raise ValueError("source_block_lengths are required when lambda_dyn > 0.")

        self.source_mean_ = source.mean(axis=0)
        self.target_mean_ = target.mean(axis=0)

        basis_aligner = build_supervised_aligner(
            "A1",
            k=max(int(self.basis_k), int(self.residual_rank)),
            reg_lambda=float(self.basis_reg_lambda),
            normalize_output=False,
        ).fit(source, y_source)
        basis = np.asarray(basis_aligner.projection_[:, : min(int(self.residual_rank), basis_aligner.projection_.shape[1])], dtype=np.float64)
        if basis.size == 0:
            basis = np.zeros((source.shape[1], 0), dtype=np.float64)
        self.basis_ = basis
        rank = int(self.basis_.shape[1])
        self.residual_matrix_ = np.zeros((rank, rank), dtype=np.float64)

        if rank == 0 or int(self.max_iter) <= 0:
            self.optimization_info_ = {"success": True, "nit": 0, "message": "Skipped optimization"}
            if float(self.lambda_dyn) > 0.0 and source_block_lengths is not None:
                self.source_operator_ = _fit_linear_state_operator_blocks(
                    _split_blocks(source, source_block_lengths),
                    ridge_alpha=float(self.ridge_alpha),
                )
            return self

        centered_source = source - self.source_mean_
        centered_target = target - self.target_mean_
        source_low_rank = centered_source @ self.basis_
        target_low_rank = centered_target @ self.basis_

        from src.models.classifiers import LDA

        lda = LDA().fit(source, y_source)
        coef = np.asarray(lda.model.coef_, dtype=np.float64)
        if coef.ndim == 1:
            coef = coef[None, :]
        score_basis = self.basis_.T @ coef.T

        source_operator = None
        if float(self.lambda_dyn) > 0.0:
            source_blocks = _split_blocks(source, source_block_lengths)
            source_operator = _fit_linear_state_operator_blocks(
                source_blocks,
                ridge_alpha=float(self.ridge_alpha),
            )
        self.source_operator_ = source_operator

        source_source_gram = source_low_rank.T @ source_low_rank / max(len(source_low_rank), 1)
        score_gram = score_basis @ score_basis.T

        target_base = centered_target + self.source_mean_
        if len(target) >= 2:
            target_prev_low_rank = target_low_rank[:-1]
            target_next_low_rank = target_low_rank[1:]
            target_prev_base = target_base[:-1]
            target_next_base = target_base[1:]
        else:
            target_prev_low_rank = np.zeros((0, rank), dtype=np.float64)
            target_next_low_rank = np.zeros((0, rank), dtype=np.float64)
            target_prev_base = np.zeros((0, source.shape[1]), dtype=np.float64)
            target_next_base = np.zeros((0, source.shape[1]), dtype=np.float64)

        kbasis = None if source_operator is None else source_operator @ self.basis_

        def _loss_and_grad(flat_s: np.ndarray) -> tuple[float, np.ndarray]:
            s_matrix = flat_s.reshape(rank, rank)

            delta_scores = source_low_rank @ s_matrix @ score_basis
            loss_cls = float(np.mean(delta_scores * delta_scores))
            grad_cls = (
                2.0 * source_source_gram @ s_matrix @ score_gram
            )

            if source_operator is not None and len(target_prev_low_rank) > 0:
                residual = target_next_base - target_prev_base @ source_operator.T
                residual = residual + target_next_low_rank @ s_matrix @ self.basis_.T
                residual = residual - target_prev_low_rank @ s_matrix @ kbasis.T
                loss_dyn = float(np.mean(residual * residual))
                grad_dyn = (
                    2.0
                    * (
                        target_next_low_rank.T @ residual @ self.basis_
                        - target_prev_low_rank.T @ residual @ kbasis
                    )
                    / len(target_prev_low_rank)
                )
            else:
                loss_dyn = 0.0
                grad_dyn = np.zeros_like(s_matrix)

            loss_reg = float(np.sum(s_matrix * s_matrix))
            grad_reg = 2.0 * s_matrix

            total_loss = (
                float(self.lambda_cls) * loss_cls
                + float(self.lambda_dyn) * loss_dyn
                + float(self.lambda_reg) * loss_reg
            )
            total_grad = (
                float(self.lambda_cls) * grad_cls
                + float(self.lambda_dyn) * grad_dyn
                + float(self.lambda_reg) * grad_reg
            )
            return total_loss, total_grad.reshape(-1)

        def _objective(flat_s: np.ndarray) -> float:
            loss, _ = _loss_and_grad(flat_s)
            return loss

        def _gradient(flat_s: np.ndarray) -> np.ndarray:
            _, grad = _loss_and_grad(flat_s)
            return grad

        init = np.zeros(rank * rank, dtype=np.float64)
        result = minimize(
            _objective,
            init,
            method="L-BFGS-B",
            jac=_gradient,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )
        self.residual_matrix_ = np.asarray(result.x.reshape(rank, rank), dtype=np.float64)
        self.optimization_info_ = {
            "success": bool(result.success),
            "nit": int(result.nit),
            "fun": float(result.fun),
            "message": str(result.message),
        }
        return self

    def transform_source(self, features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.basis_ is None or self.residual_matrix_ is None:
            raise RuntimeError("KoopmanConservativeResidualAligner is not fitted.")
        features = np.asarray(features, dtype=np.float64)
        centered = features - self.source_mean_
        residual = centered @ self.basis_ @ self.residual_matrix_ @ self.basis_.T
        return np.asarray(features + residual, dtype=np.float64)

    def transform_target(self, features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None or self.basis_ is None or self.residual_matrix_ is None:
            raise RuntimeError("KoopmanConservativeResidualAligner is not fitted.")
        features = np.asarray(features, dtype=np.float64)
        centered = features - self.target_mean_
        residual = centered @ self.basis_ @ self.residual_matrix_ @ self.basis_.T
        return np.asarray(self.source_mean_ + centered + residual, dtype=np.float64)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.transform_target(features)


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
        source_cov = _safe_cov(source_features, reg_lambda=self.eps)
        target_cov = _safe_cov(target_features, reg_lambda=self.eps)
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


@dataclass
class KoopmanIdentityAligner:
    def fit(self, source_features: np.ndarray, target_features: np.ndarray | None = None) -> "KoopmanIdentityAligner":  # noqa: ARG002
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64)


@dataclass
class KoopmanMeanShiftAligner:
    shift_strength: float = 0.5
    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "KoopmanMeanShiftAligner":
        self.source_mean_ = np.asarray(source_features, dtype=np.float64).mean(axis=0)
        self.target_mean_ = np.asarray(target_features, dtype=np.float64).mean(axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None:
            raise RuntimeError("KoopmanMeanShiftAligner is not fitted.")
        shift = float(self.shift_strength) * (self.source_mean_ - self.target_mean_)
        return np.asarray(features, dtype=np.float64) + shift


@dataclass
class KoopmanDiagonalScalingAligner:
    clip_min: float = 0.67
    clip_max: float = 1.5
    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "KoopmanDiagonalScalingAligner":
        source = np.asarray(source_features, dtype=np.float64)
        target = np.asarray(target_features, dtype=np.float64)
        self.source_mean_ = source.mean(axis=0)
        self.target_mean_ = target.mean(axis=0)
        source_std = np.clip(source.std(axis=0), 1.0e-8, None)
        target_std = np.clip(target.std(axis=0), 1.0e-8, None)
        self.scale_ = np.clip(source_std / target_std, float(self.clip_min), float(self.clip_max))
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None or self.scale_ is None:
            raise RuntimeError("KoopmanDiagonalScalingAligner is not fitted.")
        features = np.asarray(features, dtype=np.float64)
        return (features - self.target_mean_) * self.scale_ + self.source_mean_


@dataclass
class KoopmanShrinkageAffineAligner:
    rank: int = 16
    shrink: float = 0.3
    eps: float = 1.0e-6
    affine_: KoopmanAffineAligner | None = None
    basis_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "KoopmanShrinkageAffineAligner":
        source = np.asarray(source_features, dtype=np.float64)
        target = np.asarray(target_features, dtype=np.float64)
        self.affine_ = KoopmanAffineAligner(eps=self.eps).fit(source, target)
        centered = source - source.mean(axis=0)
        if centered.shape[0] > 1:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            basis = vh[: min(int(self.rank), vh.shape[0])].T
        else:
            basis = np.eye(source.shape[1], dtype=np.float64)[:, : min(int(self.rank), source.shape[1])]
        self.basis_ = np.asarray(basis, dtype=np.float64)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.affine_ is None or self.basis_ is None:
            raise RuntimeError("KoopmanShrinkageAffineAligner is not fitted.")
        features = np.asarray(features, dtype=np.float64)
        affine_features = self.affine_.transform(features)
        delta = affine_features - features
        projected_delta = delta @ self.basis_ @ self.basis_.T
        return features + float(self.shrink) * projected_delta


@dataclass
class _ProjectionAlignerBase:
    k: int = 16
    reg_lambda: float = 1.0e-3
    normalize_output: bool = False
    source_mean_: np.ndarray | None = None
    projection_: np.ndarray | None = None
    output_mean_: np.ndarray | None = None
    output_std_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, y_source: np.ndarray):  # noqa: D401
        raise NotImplementedError

    def _finalize(self, source_features: np.ndarray, primary: np.ndarray) -> None:
        source_features = np.asarray(source_features, dtype=np.float64)
        self.source_mean_ = source_features.mean(axis=0)
        centered = source_features - self.source_mean_
        self.projection_ = _complete_basis(primary, centered, self.k)
        transformed = centered @ self.projection_ @ self.projection_.T + self.source_mean_
        if self.normalize_output:
            self.output_mean_ = transformed.mean(axis=0)
            self.output_std_ = np.clip(transformed.std(axis=0), 1.0e-8, None)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.projection_ is None:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted.")
        features = np.asarray(features, dtype=np.float64)
        transformed = (features - self.source_mean_) @ self.projection_ @ self.projection_.T + self.source_mean_
        if self.normalize_output:
            if self.output_mean_ is None or self.output_std_ is None:
                raise RuntimeError(f"{self.__class__.__name__} normalization stats are missing.")
            transformed = (transformed - self.output_mean_) / self.output_std_
        return np.asarray(transformed, dtype=np.float64)


@dataclass
class KoopmanLDAAligner(_ProjectionAlignerBase):
    def fit(self, source_features: np.ndarray, y_source: np.ndarray) -> "KoopmanLDAAligner":
        source_features = np.asarray(source_features, dtype=np.float64)
        centered = source_features - source_features.mean(axis=0)
        primary = _fit_lda_directions(centered, np.asarray(y_source), self.reg_lambda)
        self._finalize(source_features, primary)
        return self


@dataclass
class KoopmanCSPAligner(_ProjectionAlignerBase):
    def fit(self, source_features: np.ndarray, y_source: np.ndarray) -> "KoopmanCSPAligner":
        source_features = np.asarray(source_features, dtype=np.float64)
        centered = source_features - source_features.mean(axis=0)
        primary = _fit_csp_directions(centered, np.asarray(y_source), self.reg_lambda)
        self._finalize(source_features, primary)
        return self


@dataclass
class KoopmanLinearAligner(_ProjectionAlignerBase):
    def fit(self, source_features: np.ndarray, y_source: np.ndarray) -> "KoopmanLinearAligner":
        source_features = np.asarray(source_features, dtype=np.float64)
        centered = source_features - source_features.mean(axis=0)
        primary = _fit_linear_directions(centered, np.asarray(y_source), self.reg_lambda)
        self._finalize(source_features, primary)
        return self


def build_supervised_aligner(
    method: str,
    *,
    k: int,
    reg_lambda: float,
    normalize_output: bool,
) -> _ProjectionAlignerBase:
    method = str(method).lower()
    if method in {"a1", "lda", "lda-style", "supervised-projection"}:
        return KoopmanLDAAligner(k=k, reg_lambda=reg_lambda, normalize_output=normalize_output)
    if method in {"a2", "csp", "csp-style", "generalized-eigen"}:
        return KoopmanCSPAligner(k=k, reg_lambda=reg_lambda, normalize_output=normalize_output)
    if method in {"a3", "linear", "linear-e2e", "end-to-end"}:
        return KoopmanLinearAligner(k=k, reg_lambda=reg_lambda, normalize_output=normalize_output)
    raise ValueError(f"Unsupported supervised aligner '{method}'")


def fit_alignment(
    source_covariances: np.ndarray,
    target_covariances: np.ndarray,
    pca_rank: int = 16,
    cov_eps: float = 1.0e-6,
    lifting: str = "quadratic",
) -> tuple[KoopmanFeatureProjector, KoopmanAffineAligner, np.ndarray, np.ndarray]:
    projector = KoopmanFeatureProjector(
        pca_rank=pca_rank, cov_eps=cov_eps, lifting=lifting
    ).fit(source_covariances)
    psi_source = projector.transform(source_covariances)
    psi_target = projector.transform(target_covariances)
    aligner = KoopmanAffineAligner(eps=cov_eps).fit(psi_source, psi_target)
    return projector, aligner, psi_source, psi_target


def transform(
    covariances: np.ndarray,
    projector: KoopmanFeatureProjector,
    aligner: KoopmanAffineAligner | _ProjectionAlignerBase | None = None,
) -> np.ndarray:
    psi = projector.transform(covariances)
    return psi if aligner is None else aligner.transform(psi)
