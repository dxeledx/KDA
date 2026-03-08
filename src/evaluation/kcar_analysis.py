from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score


def _sym_to_vec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    values: List[float] = []
    sqrt2 = float(np.sqrt(2.0))
    for row in range(n):
        values.append(float(matrix[row, row]))
        for col in range(row + 1, n):
            values.append(sqrt2 * float(matrix[row, col]))
    return np.asarray(values, dtype=np.float64)


def _matrix_power_spd(matrix: np.ndarray, power: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    eigvals, eigvecs = eigh(0.5 * (matrix + matrix.T))
    eigvals = np.clip(eigvals, 1.0e-12, None)
    return eigvecs @ np.diag(eigvals**power) @ eigvecs.T


def _matrix_log_spd(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    eigvals, eigvecs = eigh(0.5 * (matrix + matrix.T))
    eigvals = np.clip(eigvals, 1.0e-12, None)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def _lift_quadratic(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    return np.concatenate([states, states * states, np.ones((states.shape[0], 1))], axis=1)


@dataclass
class TangentProjector:
    reference_covariance: np.ndarray
    pca: Optional[PCA]

    def transform(self, covariances: np.ndarray) -> np.ndarray:
        covariances = np.asarray(covariances, dtype=np.float64)
        ref_inv_sqrt = _matrix_power_spd(self.reference_covariance, -0.5)
        vectors = []
        for covariance in covariances:
            whitened = ref_inv_sqrt @ covariance @ ref_inv_sqrt
            tangent = _matrix_log_spd(whitened)
            vectors.append(_sym_to_vec(tangent))
        tangent_vectors = np.asarray(vectors, dtype=np.float64)
        if self.pca is not None:
            return self.pca.transform(tangent_vectors)
        return tangent_vectors


@dataclass
class KoopmanOperator:
    matrix: np.ndarray

    def transform(self, states: np.ndarray) -> np.ndarray:
        return _lift_quadratic(states)

    def predict_lifted(self, states: np.ndarray) -> np.ndarray:
        lifted = self.transform(states)
        return (self.matrix @ lifted.T).T


def fit_tangent_projector(
    source_covariances: np.ndarray,
    reference_covariance: np.ndarray,
    pca_rank: int = 16,
) -> TangentProjector:
    source_covariances = np.asarray(source_covariances, dtype=np.float64)
    projector = TangentProjector(
        reference_covariance=np.asarray(reference_covariance, dtype=np.float64),
        pca=None,
    )
    tangent = projector.transform(source_covariances)
    if tangent.shape[1] <= pca_rank:
        return projector

    pca = PCA(n_components=min(int(pca_rank), tangent.shape[0], tangent.shape[1]))
    pca.fit(tangent)
    return TangentProjector(reference_covariance=projector.reference_covariance, pca=pca)


def fit_koopman_operator(states: np.ndarray, ridge_alpha: float = 1.0e-3) -> KoopmanOperator:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("Need at least two states to fit a Koopman operator.")

    lifted_prev = _lift_quadratic(states[:-1])
    lifted_next = _lift_quadratic(states[1:])
    gram = lifted_prev.T @ lifted_prev
    ridge = float(ridge_alpha) * np.eye(gram.shape[0], dtype=np.float64)
    operator = (lifted_next.T @ lifted_prev) @ np.linalg.pinv(gram + ridge)
    return KoopmanOperator(matrix=np.asarray(operator, dtype=np.float64))


def fit_subjectwise_global_koopman(
    state_blocks: Sequence[np.ndarray],
    ridge_alpha: float = 1.0e-3,
) -> KoopmanOperator:
    lifted_prev_blocks, lifted_next_blocks = [], []
    for states in state_blocks:
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[0] < 2:
            continue
        lifted_prev_blocks.append(_lift_quadratic(states[:-1]))
        lifted_next_blocks.append(_lift_quadratic(states[1:]))

    if not lifted_prev_blocks:
        raise ValueError("No valid state blocks to fit global Koopman operator.")

    lifted_prev = np.concatenate(lifted_prev_blocks, axis=0)
    lifted_next = np.concatenate(lifted_next_blocks, axis=0)
    gram = lifted_prev.T @ lifted_prev
    ridge = float(ridge_alpha) * np.eye(gram.shape[0], dtype=np.float64)
    operator = (lifted_next.T @ lifted_prev) @ np.linalg.pinv(gram + ridge)
    return KoopmanOperator(matrix=np.asarray(operator, dtype=np.float64))


def compute_transition_residuals(
    states: np.ndarray,
    operator: KoopmanOperator,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)

    lifted_target = operator.transform(states[1:])
    lifted_pred = operator.predict_lifted(states[:-1])
    return np.linalg.norm(lifted_target - lifted_pred, axis=1)


def compute_kcar(
    source_residuals: np.ndarray,
    target_residuals: np.ndarray,
    eps: float = 1.0e-8,
) -> float:
    source_residuals = np.asarray(source_residuals, dtype=np.float64)
    target_residuals = np.asarray(target_residuals, dtype=np.float64)
    if source_residuals.shape != target_residuals.shape:
        raise ValueError("Residual arrays must have the same shape.")
    if source_residuals.size == 0:
        return 0.0

    ratio = (source_residuals - target_residuals) / (
        source_residuals + target_residuals + float(eps)
    )
    return float(np.clip(np.mean(ratio), -1.0, 1.0))


def label_window_alignment_risk(
    acc_ra: float,
    acc_w0: float,
    acc_w05: float,
    window_size: int = 32,
) -> Dict[str, float | str]:
    threshold = 1.0 / float(window_size)
    best_deviation = max(float(acc_w0), float(acc_w05))
    delta_dev_vs_ra = best_deviation - float(acc_ra)

    if delta_dev_vs_ra >= threshold:
        label = "deviation-beneficial"
    elif -delta_dev_vs_ra >= threshold:
        label = "ra-safe"
    else:
        label = "neutral"

    return {
        "label": label,
        "delta_dev_vs_ra": float(delta_dev_vs_ra),
        "best_deviation_accuracy": float(best_deviation),
    }


def _risk_direction(column: str) -> float:
    if column == "conf_max":
        return -1.0
    return 1.0


def _safe_metric_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def compare_window_scores(
    window_metrics: pd.DataFrame,
    score_columns: Sequence[str],
    positive_label: str = "deviation-beneficial",
    negative_label: str = "ra-safe",
) -> Tuple[pd.DataFrame, Dict]:
    required = {"subject", "label", "delta_dev_vs_ra", *score_columns}
    missing = required.difference(window_metrics.columns)
    if missing:
        raise ValueError(f"window_metrics missing required columns: {sorted(missing)}")

    rows: List[Dict] = []
    eligible = window_metrics[window_metrics["label"].isin([positive_label, negative_label])].copy()
    if eligible.empty:
        raise ValueError("No eligible windows for KCAR comparison.")

    for subject, subject_df in window_metrics.groupby("subject"):
        eligible_subject = subject_df[subject_df["label"].isin([positive_label, negative_label])].copy()
        y_true = (eligible_subject["label"] == positive_label).astype(np.int64).to_numpy()
        for score in score_columns:
            risk_scores = _risk_direction(score) * eligible_subject[score].to_numpy(dtype=np.float64)
            if len(np.unique(y_true)) < 2 or np.allclose(risk_scores, risk_scores[0]):
                auroc = float("nan")
                auprc = float("nan")
            else:
                auroc = float(roc_auc_score(y_true, risk_scores))
                auprc = float(average_precision_score(y_true, risk_scores))

            correlation_scores = _risk_direction(score) * subject_df[score].to_numpy(dtype=np.float64)
            spearman = float(
                spearmanr(correlation_scores, subject_df["delta_dev_vs_ra"].to_numpy(dtype=np.float64)).statistic
            )
            rows.append(
                {
                    "subject": int(subject),
                    "score": score,
                    "auroc": auroc,
                    "auprc": auprc,
                    "spearman": spearman,
                    "eligible_windows": int(len(eligible_subject)),
                }
            )

    comparison_df = pd.DataFrame(rows)
    kcar_df = comparison_df[comparison_df["score"] == "rho_kcar"]
    subject_wins = {}
    for score in score_columns:
        if score == "rho_kcar":
            continue
        merged = kcar_df[["subject", "auroc"]].merge(
            comparison_df[comparison_df["score"] == score][["subject", "auroc"]],
            on="subject",
            suffixes=("_kcar", "_baseline"),
            how="inner",
        )
        subject_wins[score] = int(np.sum(merged["auroc_kcar"] > merged["auroc_baseline"]))

    summary = {
        "mean_auroc": _safe_metric_mean(kcar_df["auroc"].tolist()),
        "mean_auprc": _safe_metric_mean(kcar_df["auprc"].tolist()),
        "mean_spearman": _safe_metric_mean(kcar_df["spearman"].tolist()),
        "subject_wins_vs_heuristics": subject_wins,
        "eligible_window_count": int(len(eligible)),
    }
    return comparison_df.sort_values(["subject", "score"]).reset_index(drop=True), summary
