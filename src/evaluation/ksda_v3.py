from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.alignment.koopman_alignment import (
    KoopmanAffineAligner,
    KoopmanDiagonalScalingAligner,
    KoopmanIdentityAligner,
    KoopmanMeanShiftAligner,
    KoopmanShrinkageAffineAligner,
    build_supervised_aligner,
)
from src.evaluation.kcar_analysis import compute_transition_residuals
from src.features.covariance import compute_covariances


@dataclass
class KSDAV3Fold:
    target_subject: int
    cov_source: np.ndarray
    y_source: np.ndarray
    cov_target_train: np.ndarray
    cov_target_test: np.ndarray
    y_target_test: np.ndarray
    source_block_lengths: List[int]


def build_window_slices(total_trials: int, window_size: int) -> List[tuple[int, int]]:
    total_trials = int(total_trials)
    window_size = int(window_size)
    if total_trials <= 0 or window_size <= 0:
        return []
    return [(start, min(start + window_size, total_trials)) for start in range(0, total_trials, window_size)]


def oracle_usage_stats(actions: np.ndarray) -> Dict[str, float | int]:
    actions = np.asarray(actions)
    if actions.size == 0:
        return {"most_common_ratio": 0.0, "unique_count": 0}
    unique, counts = np.unique(actions, return_counts=True)
    return {
        "most_common_ratio": float(counts.max() / counts.sum()),
        "unique_count": int(unique.size),
    }


def compute_window_oracle_actions(window_scores: np.ndarray) -> tuple[np.ndarray, float]:
    window_scores = np.asarray(window_scores, dtype=np.float64)
    if window_scores.ndim != 2:
        raise ValueError(f"Expected 2D window_scores, got {window_scores.shape}")
    chosen = np.argmax(window_scores, axis=1).astype(np.int64)
    oracle_acc = float(np.mean(window_scores[np.arange(window_scores.shape[0]), chosen])) if window_scores.shape[0] > 0 else 0.0
    return chosen, oracle_acc


def expand_window_actions_to_trials(actions: np.ndarray, total_trials: int, window_size: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    expanded = np.zeros(int(total_trials), dtype=np.float64)
    for window_idx, (start, end) in enumerate(build_window_slices(total_trials, window_size)):
        expanded[start:end] = actions[window_idx]
    return expanded


def load_ksda_v3_folds(loader, all_subjects: Sequence[int], target_subjects: Sequence[int], pre, cov_eps: float) -> List[KSDAV3Fold]:
    folds: List[KSDAV3Fold] = []
    for target_subject in target_subjects:
        X_source_blocks, y_source_blocks = [], []
        for subject in all_subjects:
            if int(subject) == int(target_subject):
                continue
            X_train_subject, y_train_subject = loader.load_subject(int(subject), split="train")
            X_source_blocks.append(X_train_subject)
            y_source_blocks.append(y_train_subject)

        X_source_raw = np.concatenate(X_source_blocks, axis=0)
        y_source = np.concatenate(y_source_blocks, axis=0)
        X_target_train_raw, _ = loader.load_subject(int(target_subject), split="train")
        X_target_test_raw, y_target_test = loader.load_subject(int(target_subject), split="test")

        X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
        X_target_train = pre.transform(X_target_train_raw)
        X_target_test = pre.transform(X_target_test_raw)

        folds.append(
            KSDAV3Fold(
                target_subject=int(target_subject),
                cov_source=compute_covariances(X_source, eps=cov_eps),
                y_source=np.asarray(y_source, dtype=np.int64),
                cov_target_train=compute_covariances(X_target_train, eps=cov_eps),
                cov_target_test=compute_covariances(X_target_test, eps=cov_eps),
                y_target_test=np.asarray(y_target_test, dtype=np.int64),
                source_block_lengths=[block.shape[0] for block in X_source_blocks],
            )
        )
    return folds


def build_local_expert_aligners(
    psi_source: np.ndarray,
    psi_target_train: np.ndarray,
    y_source: np.ndarray,
    cov_eps: float = 1.0e-6,
) -> Dict[str, object]:
    experts = {
        "E0_identity": KoopmanIdentityAligner().fit(psi_source),
        "E1_mean_shift": KoopmanMeanShiftAligner(shift_strength=0.5).fit(psi_source, psi_target_train),
        "E2_diag_scaling": KoopmanDiagonalScalingAligner().fit(psi_source, psi_target_train),
        "E3_low_rank_shrinkage": KoopmanShrinkageAffineAligner(rank=16, shrink=0.3, eps=cov_eps).fit(
            psi_source, psi_target_train
        ),
        "E4_supervised_subspace": build_supervised_aligner(
            "A1", k=32, reg_lambda=1.0e-4, normalize_output=True
        ).fit(psi_source, y_source),
    }
    return experts


def apply_expert_aligner(aligner, psi: np.ndarray) -> np.ndarray:
    return np.asarray(aligner.transform(np.asarray(psi, dtype=np.float64)), dtype=np.float64)


def summarize_oracle_gate(
    oracle_scores: np.ndarray,
    action_labels: Sequence,
) -> Dict[str, object]:
    chosen, oracle_acc = compute_window_oracle_actions(oracle_scores)
    chosen_labels = np.asarray([action_labels[int(idx)] for idx in chosen], dtype=object)
    stats = oracle_usage_stats(chosen_labels)
    return {
        "chosen_indices": chosen,
        "chosen_labels": chosen_labels,
        "oracle_accuracy": oracle_acc,
        **stats,
    }


def paired_wins(left: Iterable[float], right: Iterable[float]) -> Dict[str, int | float]:
    left = np.asarray(list(left), dtype=np.float64)
    right = np.asarray(list(right), dtype=np.float64)
    delta = left - right
    return {
        "mean_delta": float(delta.mean()) if delta.size else 0.0,
        "wins": int(np.sum(delta > 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "draws": int(np.sum(np.isclose(delta, 0.0))),
    }


def fit_linear_multiclass_selector(X: np.ndarray, y: np.ndarray, num_classes: int, reg_lambda: float = 1.0e-3) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mean = X.mean(axis=0)
    std = np.clip(X.std(axis=0), 1.0e-8, None)
    Xn = (X - mean) / std
    targets = np.eye(int(num_classes), dtype=np.float64)[y]
    gram = Xn.T @ Xn + float(reg_lambda) * np.eye(Xn.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, Xn.T @ targets)
    return {"mean": mean, "std": std, "weights": weights}


def predict_linear_multiclass_selector(model: Dict[str, np.ndarray], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Xn = (X - model["mean"]) / model["std"]
    scores = Xn @ model["weights"]
    labels = np.argmax(scores, axis=1).astype(np.int64)
    return labels, scores


def fit_linear_scalar_proxy(X: np.ndarray, y: np.ndarray, reg_lambda: float = 1.0e-3) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = X.mean(axis=0)
    std = np.clip(X.std(axis=0), 1.0e-8, None)
    Xn = (X - mean) / std
    gram = Xn.T @ Xn + float(reg_lambda) * np.eye(Xn.shape[1], dtype=np.float64)
    beta = np.linalg.solve(gram, Xn.T @ y)
    return {"mean": mean, "std": std, "beta": beta}


def predict_linear_scalar_proxy(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xn = (X - model["mean"]) / model["std"]
    return Xn @ model["beta"]


def compute_window_feature_matrix(
    z_states: np.ndarray,
    psi_states: np.ndarray,
    source_operator,
    target_operator,
    lda_raw,
    window_size: int,
    source_mean_z: np.ndarray,
    source_diag_var_z: np.ndarray,
) -> Dict[str, np.ndarray]:
    z_states = np.asarray(z_states, dtype=np.float64)
    psi_states = np.asarray(psi_states, dtype=np.float64)
    source_mean_z = np.asarray(source_mean_z, dtype=np.float64)
    source_diag_var_z = np.asarray(source_diag_var_z, dtype=np.float64)
    rows = []
    window_ids = []
    for window_idx, (start, end) in enumerate(build_window_slices(len(z_states), window_size)):
        z_window = z_states[start:end]
        psi_window = psi_states[start:end]
        if len(z_window) >= 2:
            r_src = float(np.mean(compute_transition_residuals(z_window, source_operator)))
            r_tgt = float(np.mean(compute_transition_residuals(z_window, target_operator)))
            step_smooth = float(np.mean(np.linalg.norm(np.diff(z_window, axis=0), axis=1)))
        else:
            r_src = 0.0
            r_tgt = 0.0
            step_smooth = 0.0
        delta_r = float(r_tgt - r_src)

        prob = np.asarray(lda_raw.predict_proba(psi_window), dtype=np.float64)
        sorted_prob = np.sort(prob, axis=1)
        if sorted_prob.shape[1] >= 2:
            margin = float(np.mean(sorted_prob[:, -1] - sorted_prob[:, -2]))
        else:
            margin = float(np.mean(sorted_prob[:, -1]))
        pred = np.argmax(prob, axis=1)
        stability = float(np.mean(pred[1:] == pred[:-1])) if len(pred) > 1 else 1.0

        mean_drift = float(np.linalg.norm(z_window.mean(axis=0) - source_mean_z))
        if len(z_window) > 1:
            window_diag_var = np.var(z_window, axis=0)
        else:
            window_diag_var = np.zeros_like(source_diag_var_z)
        cov_drift = float(np.linalg.norm(window_diag_var - source_diag_var_z))
        drift = float(mean_drift + 0.1 * cov_drift)

        rows.append(np.asarray([r_src, r_tgt, delta_r, margin, drift, stability], dtype=np.float64))
        window_ids.append(window_idx)

    return {
        "features": np.asarray(rows, dtype=np.float64),
        "window_id": np.asarray(window_ids, dtype=np.int64),
    }


def build_temporary_endpoint_aligner(
    method: str,
    psi_source: np.ndarray,
    psi_target_train: np.ndarray,
    y_source: np.ndarray,
    *,
    k: int | None,
    reg_lambda: float | None,
    normalize_output: bool,
    cov_eps: float = 1.0e-6,
):
    method = str(method)
    if method == "A0":
        return KoopmanIdentityAligner().fit(psi_source)
    if method == "legacy-affine":
        return KoopmanAffineAligner(eps=cov_eps).fit(psi_source, psi_target_train)
    return build_supervised_aligner(
        method,
        k=int(k if k is not None else 16),
        reg_lambda=float(reg_lambda if reg_lambda is not None else 1.0e-3),
        normalize_output=bool(normalize_output),
    ).fit(psi_source, y_source)
